package ndvek

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/viterin/vek"
	"github.com/viterin/vek/vek32"
)

type DType int

const (
	Float64 DType = iota
	Float32
	Bool
)

// NdArray represents a multi-dimensional array with shape and data.
type NdArray struct {
	shape []int
	data  any // []float64, []float32, or []bool
	dtype DType
}

func (a *NdArray) DType() DType {
	return a.dtype
}

// Float64Data returns the underlying []float64 data, or nil if the dtype is not Float64.
func (a *NdArray) Float64Data() []float64 {
	if a.dtype == Float64 {
		return a.data.([]float64)
	}
	return nil
}

// Float32Data returns the underlying []float32 data, or nil if the dtype is not Float32.
func (a *NdArray) Float32Data() []float32 {
	if a.dtype == Float32 {
		return a.data.([]float32)
	}
	return nil
}

// BoolData returns the underlying []bool data, or nil if the dtype is not Bool.
func (a *NdArray) BoolData() []bool {
	if a.dtype == Bool {
		return a.data.([]bool)
	}
	return nil
}

// NewNdArray creates a new NdArray given a shape and initial data.
// Data can be []float64, []float32, or []bool.
func NewNdArray(shape []int, data any) (*NdArray, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)

	var dtype DType

	switch v := data.(type) {
	case []float64:
		if size != len(v) {
			return nil, errors.New("data length does not match shape dimensions")
		}
		dtype = Float64
	case []float32:
		if size != len(v) {
			return nil, errors.New("data length does not match shape dimensions")
		}
		dtype = Float32
	case []bool:
		if size != len(v) {
			return nil, errors.New("data length does not match shape dimensions")
		}
		dtype = Bool
	default:
		return nil, errors.New("unsupported data type")
	}

	return &NdArray{shape: shapeCopy, data: data, dtype: dtype}, nil
}

// broadcastShapes finds a broadcasted shape from two shapes.
func broadcastShapes(shape1, shape2 []int) ([]int, error) {
	len1, len2 := len(shape1), len(shape2)
	maxLen := max(len2, len1)
	broadcastedShape := make([]int, maxLen)

	for i := range maxLen {
		dim1, dim2 := 1, 1
		if i < len1 {
			dim1 = shape1[len1-1-i]
		}
		if i < len2 {
			dim2 = shape2[len2-1-i]
		}
		if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
			return nil, fmt.Errorf("incompatible shapes for broadcasting: %v and %v", shape1, shape2)
		}
		broadcastedShape[maxLen-1-i] = max(dim1, dim2)
	}
	return broadcastedShape, nil
}

// broadcastIndex converts a single index to the correct offset for broadcasting.
func broadcastIndex(shape, broadcastShape []int, index int) (int, error) {
	if len(shape) > len(broadcastShape) {
		return 0, errors.New("original shape cannot be larger than broadcast shape")
	}

	originalIndex := 0
	stride := 1

	for i := len(broadcastShape) - 1; i >= 0; i-- {
		broadcastDim := broadcastShape[i]
		shapeDim := 1
		if i-len(broadcastShape)+len(shape) >= 0 {
			shapeDim = shape[i-len(broadcastShape)+len(shape)]
		}

		currentCoord := index % broadcastDim
		index /= broadcastDim

		if shapeDim != broadcastDim && shapeDim == 1 {
			currentCoord = 0
		}

		originalIndex += currentCoord * stride
		stride *= shapeDim
	}

	return originalIndex, nil
}

func (a *NdArray) ApplyHadamardOp(op func(float64) float64) error {
	if a.dtype == Bool {
		return errors.New("ApplyHadamardOp not supported for Bool arrays")
	}
	if a.dtype == Float32 {
		f32Data := a.data.([]float32)
		f64Data := make([]float64, len(f32Data))
		for i, v := range f32Data {
			f64Data[i] = op(float64(v))
		}
		a.data = f64Data
		a.dtype = Float64
		return nil
	}

	d := a.data.([]float64)
	for i := range d {
		d[i] = op(d[i])
	}
	return nil
}

// ApplyOp applies an arithmetic operation with broadcasting.
// Returns a Float64 array because op returns float64.
func ApplyOp(a, b *NdArray, op func(float64, float64) float64) (*NdArray, error) {
	bShape, err := broadcastShapes(a.shape, b.shape)
	if err != nil {
		return nil, err
	}

	sizeOf := ProdInt(bShape)
	resultData := make([]float64, sizeOf)
	result := &NdArray{shape: bShape, data: resultData, dtype: Float64}

	aData, err := a.toFloat64()
	if err != nil {
		return nil, err
	}
	bData, err := b.toFloat64()
	if err != nil {
		return nil, err
	}

	for i := range resultData {
		aIndex, err := broadcastIndex(a.shape, bShape, i)
		if err != nil {
			return nil, err
		}
		bIndex, err := broadcastIndex(b.shape, bShape, i)
		if err != nil {
			return nil, err
		}
		resultData[i] = op(aData[aIndex], bData[bIndex])
	}
	return result, nil
}

// Add performs element-wise addition with broadcasting.
func Add(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.Add(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.Add(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.AddNumber(a.data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.AddNumber(a.mustFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: b.shape, data: vek32.AddNumber(b.data.([]float32), float32(aVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: b.shape, data: vek.AddNumber(b.mustFloat64(), aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x + y })
}

// Subtract performs element-wise subtraction with broadcasting.
func Subtract(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.Sub(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.Sub(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.SubNumber(a.data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.SubNumber(a.mustFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			diff := vek32.SubNumber(b.data.([]float32), float32(aVal))
			return &NdArray{shape: b.shape, data: vek32.MulNumber(diff, -1), dtype: Float32}, nil
		}
		diff := vek.SubNumber(b.mustFloat64(), aVal)
		return &NdArray{shape: b.shape, data: vek.MulNumber(diff, -1), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x - y })
}

// Multiply performs element-wise multiplication with broadcasting.
func Multiply(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.Mul(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.Mul(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.MulNumber(a.data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.MulNumber(a.mustFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: b.shape, data: vek32.MulNumber(b.data.([]float32), float32(aVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: b.shape, data: vek.MulNumber(b.mustFloat64(), aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x * y })
}

// Divide performs element-wise division with broadcasting.
func Divide(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.Div(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.Div(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.DivNumber(a.data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.DivNumber(a.mustFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			inv := vek32.Inv(b.data.([]float32))
			return &NdArray{shape: b.shape, data: vek32.MulNumber(inv, float32(aVal)), dtype: Float32}, nil
		}
		inv := vek.Inv(b.mustFloat64())
		return &NdArray{shape: b.shape, data: vek.MulNumber(inv, aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x / y })
}

// Pow performs element-wise exponentiation with broadcasting.
func Pow(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, data: vek32.Pow(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, data: vek.Pow(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return math.Pow(x, y) })
}

// Minimum performs element-wise minimum.
func Minimum(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("shapes must be equal for Minimum")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Minimum(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
	}
	return &NdArray{shape: a.shape, data: vek.Minimum(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
}

// Maximum performs element-wise maximum.
func Maximum(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("shapes must be equal for Maximum")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Maximum(a.data.([]float32), b.data.([]float32)), dtype: Float32}, nil
	}
	return &NdArray{shape: a.shape, data: vek.Maximum(a.mustFloat64(), b.mustFloat64()), dtype: Float64}, nil
}

// Shape returns the shape of the ndarray.
func (a *NdArray) Shape() []int {
	return a.shape
}

func ProdInt(x []int) int {
	out := 1
	for _, y := range x {
		out *= y
	}
	return out
}

func Zeros(shape []int) *NdArray {
	size := ProdInt(shape)
	data := make([]float64, size)
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &NdArray{shape: shapeCopy, data: data, dtype: Float64}
}

// Ones creates a new float64 NdArray filled with ones.
func Ones(shape []int) *NdArray {
	size := ProdInt(shape)
	data := vek.Ones(size)
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &NdArray{shape: shapeCopy, data: data, dtype: Float64}
}

func (a *NdArray) AddScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.AddNumber(a.data.([]float32), float32(b)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.AddNumber(a.mustFloat64(), b), dtype: Float64}
}

func (a *NdArray) SubScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.SubNumber(a.data.([]float32), float32(b)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.SubNumber(a.mustFloat64(), b), dtype: Float64}
}

func (a *NdArray) MulScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.MulNumber(a.data.([]float32), float32(b)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.MulNumber(a.mustFloat64(), b), dtype: Float64}
}

func (a *NdArray) DivScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.DivNumber(a.data.([]float32), float32(b)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.DivNumber(a.mustFloat64(), b), dtype: Float64}
}

func (x *NdArray) InsertAxis(pos int) (*NdArray, error) {
	rank := len(x.shape)
	if pos < 0 {
		pos += rank + 1
	}

	result := make([]int, rank+1)
	copy(result[:pos], (x.shape)[:pos])
	result[pos] = 1
	copy(result[pos+1:], x.shape[pos:])

	return NewNdArray(result, x.data)
}

func (x *NdArray) Reshape(shape []int) (*NdArray, error) {
	if ProdInt(shape) != ProdInt(x.shape) {
		return nil, fmt.Errorf("cannot reshape array of size %d into shape %v (size %d)", ProdInt(x.shape), shape, ProdInt(shape))
	}
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	x.shape = shapeCopy
	return x, nil
}

// toFloat64 converts numeric data to []float64, returning an error for Bool arrays.
func (a *NdArray) toFloat64() ([]float64, error) {
	switch a.dtype {
	case Float64:
		return a.data.([]float64), nil
	case Float32:
		return vek.FromFloat32(a.data.([]float32)), nil
	default:
		return nil, errors.New("cannot convert Bool array to float64")
	}
}

// mustFloat64 converts numeric data to []float64. Panics on Bool arrays.
// Use only in code paths where dtype has already been checked.
func (a *NdArray) mustFloat64() []float64 {
	d, err := a.toFloat64()
	if err != nil {
		panic(err)
	}
	return d
}

func (a *NdArray) toFloat32() ([]float32, error) {
	switch a.dtype {
	case Float32:
		return a.data.([]float32), nil
	case Float64:
		return vek.ToFloat32(a.data.([]float64)), nil
	default:
		return nil, errors.New("cannot convert Bool array to float32")
	}
}

func (a *NdArray) Get(index []int) (float64, error) {
	if len(index) != len(a.shape) {
		return 0, errors.New("index length does not match array dimensions")
	}

	offset := 0
	for i, coord := range index {
		if coord < 0 || coord >= a.shape[i] {
			return 0, errors.New("index out of bounds")
		}
		offset = offset*a.shape[i] + coord
	}

	switch a.dtype {
	case Float64:
		return a.data.([]float64)[offset], nil
	case Float32:
		return float64(a.data.([]float32)[offset]), nil
	default:
		return 0, errors.New("Get not supported for Bool arrays; use BoolData()")
	}
}

func Linspace(start, stop float64, numPoints int) []float64 {
	if numPoints <= 0 {
		return nil
	}
	if numPoints == 1 {
		return []float64{start}
	}

	step := (stop - start) / float64(numPoints-1)
	result := make([]float64, numPoints)
	for i := range numPoints {
		result[i] = start + float64(i)*step
	}
	return result
}

// --- Aggregation operations (SIMD-backed) ---

func (a *NdArray) Sum() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Sum(a.data.([]float32)))
	}
	return vek.Sum(a.mustFloat64())
}

func (a *NdArray) Mean() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Mean(a.data.([]float32)))
	}
	return vek.Mean(a.mustFloat64())
}

func (a *NdArray) Min() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Min(a.data.([]float32)))
	}
	return vek.Min(a.mustFloat64())
}

func (a *NdArray) Max() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Max(a.data.([]float32)))
	}
	return vek.Max(a.mustFloat64())
}

func (a *NdArray) Prod() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Prod(a.data.([]float32)))
	}
	return vek.Prod(a.mustFloat64())
}

// Median returns the median value.
func (a *NdArray) Median() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Median(a.data.([]float32)))
	}
	return vek.Median(a.mustFloat64())
}

// Quantile returns the q-th quantile (0 <= q <= 1).
func (a *NdArray) Quantile(q float64) float64 {
	if a.dtype == Float32 {
		return float64(vek32.Quantile(a.data.([]float32), float32(q)))
	}
	return vek.Quantile(a.mustFloat64(), q)
}

// ArgMin returns the index of the minimum element.
func (a *NdArray) ArgMin() int {
	if a.dtype == Float32 {
		return vek32.ArgMin(a.data.([]float32))
	}
	return vek.ArgMin(a.mustFloat64())
}

// ArgMax returns the index of the maximum element.
func (a *NdArray) ArgMax() int {
	if a.dtype == Float32 {
		return vek32.ArgMax(a.data.([]float32))
	}
	return vek.ArgMax(a.mustFloat64())
}

// --- Unary operations (SIMD-backed) ---

func (a *NdArray) Abs() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Abs(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Abs(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) Neg() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Neg(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Neg(a.mustFloat64()), dtype: Float64}
}

// Inv computes element-wise reciprocal (1/x).
func (a *NdArray) Inv() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Inv(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Inv(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) Sqrt() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Sqrt(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Sqrt(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) Round() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Round(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Round(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) Floor() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Floor(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Floor(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) Ceil() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Ceil(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.Ceil(a.mustFloat64()), dtype: Float64}
}

// --- Transcendental functions (vek32 SIMD-backed for Float32, math stdlib for Float64) ---

// Sin computes element-wise sine.
func (a *NdArray) Sin() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Sin(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Sin(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// Cos computes element-wise cosine.
func (a *NdArray) Cos() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Cos(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Cos(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// Exp computes element-wise exponential (e^x).
func (a *NdArray) Exp() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Exp(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Exp(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// Log computes element-wise natural logarithm.
func (a *NdArray) Log() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Log(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Log(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// Log2 computes element-wise base-2 logarithm.
func (a *NdArray) Log2() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Log2(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Log2(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// Log10 computes element-wise base-10 logarithm.
func (a *NdArray) Log10() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.Log10(a.data.([]float32)), dtype: Float32}
	}
	d := a.mustFloat64()
	out := make([]float64, len(d))
	for i, v := range d {
		out[i] = math.Log10(v)
	}
	return &NdArray{shape: a.shape, data: out, dtype: Float64}
}

// --- Cumulative operations (SIMD-backed) ---

func (a *NdArray) CumSum() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.CumSum(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.CumSum(a.mustFloat64()), dtype: Float64}
}

func (a *NdArray) CumProd() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, data: vek32.CumProd(a.data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, data: vek.CumProd(a.mustFloat64()), dtype: Float64}
}

// --- Vector operations (SIMD-backed) ---

// Dot computes the dot product of two 1-D arrays.
func Dot(a, b *NdArray) (float64, error) {
	if len(a.shape) != 1 || len(b.shape) != 1 {
		return 0, errors.New("Dot requires 1-D arrays")
	}
	if a.shape[0] != b.shape[0] {
		return 0, errors.New("Dot requires arrays of equal length")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return float64(vek32.Dot(a.data.([]float32), b.data.([]float32))), nil
	}
	return vek.Dot(a.mustFloat64(), b.mustFloat64()), nil
}

// Norm computes the Euclidean (L2) norm.
func (a *NdArray) Norm() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Norm(a.data.([]float32)))
	}
	return vek.Norm(a.mustFloat64())
}

// ManhattanNorm computes the L1 norm.
func (a *NdArray) ManhattanNorm() float64 {
	if a.dtype == Float32 {
		return float64(vek32.ManhattanNorm(a.data.([]float32)))
	}
	return vek.ManhattanNorm(a.mustFloat64())
}

// Distance computes the Euclidean distance between two arrays.
func Distance(a, b *NdArray) (float64, error) {
	if !shapesEqual(a.shape, b.shape) {
		return 0, errors.New("shapes must be equal for Distance")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return float64(vek32.Distance(a.data.([]float32), b.data.([]float32))), nil
	}
	return vek.Distance(a.mustFloat64(), b.mustFloat64()), nil
}

// ManhattanDistance computes the L1 distance between two arrays.
func ManhattanDistance(a, b *NdArray) (float64, error) {
	if !shapesEqual(a.shape, b.shape) {
		return 0, errors.New("shapes must be equal for ManhattanDistance")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return float64(vek32.ManhattanDistance(a.data.([]float32), b.data.([]float32))), nil
	}
	return vek.ManhattanDistance(a.mustFloat64(), b.mustFloat64()), nil
}

// CosineSimilarity computes the cosine similarity between two arrays.
func CosineSimilarity(a, b *NdArray) (float64, error) {
	if !shapesEqual(a.shape, b.shape) {
		return 0, errors.New("shapes must be equal for CosineSimilarity")
	}
	if a.dtype == Float32 && b.dtype == Float32 {
		return float64(vek32.CosineSimilarity(a.data.([]float32), b.data.([]float32))), nil
	}
	return vek.CosineSimilarity(a.mustFloat64(), b.mustFloat64()), nil
}

// --- Comparison operations (SIMD-backed) ---

// Eq performs element-wise equality comparison.
func Eq(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Eq_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Eq_Into(data, a.data.([]float32), b.data.([]float32))
	} else if a.dtype == Bool && b.dtype == Bool {
		aData := a.data.([]bool)
		bData := b.data.([]bool)
		for i := range size {
			data[i] = aData[i] == bData[i]
		}
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Eq_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Neq performs element-wise non-equality comparison.
func Neq(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Neq_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Neq_Into(data, a.data.([]float32), b.data.([]float32))
	} else if a.dtype == Bool && b.dtype == Bool {
		aData := a.data.([]bool)
		bData := b.data.([]bool)
		for i := range size {
			data[i] = aData[i] != bData[i]
		}
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Neq_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Lt performs element-wise less than comparison.
func Lt(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	data := make([]bool, ProdInt(a.shape))

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Lt_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Lt_Into(data, a.data.([]float32), b.data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Lt_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Lte performs element-wise less than or equal comparison.
func Lte(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	data := make([]bool, ProdInt(a.shape))

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Lte_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Lte_Into(data, a.data.([]float32), b.data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Lte_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Gt performs element-wise greater than comparison.
func Gt(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	data := make([]bool, ProdInt(a.shape))

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Gt_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Gt_Into(data, a.data.([]float32), b.data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Gt_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Gte performs element-wise greater than or equal comparison.
func Gte(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	data := make([]bool, ProdInt(a.shape))

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Gte_Into(data, a.data.([]float64), b.data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Gte_Into(data, a.data.([]float32), b.data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Gte_Into(data, a.mustFloat64(), b.mustFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// --- Boolean operations (SIMD-backed) ---

// And performs element-wise logical AND.
func And(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	if a.dtype != Bool || b.dtype != Bool {
		return nil, errors.New("logical operations require boolean arrays")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)
	vek.And_Into(data, a.data.([]bool), b.data.([]bool))
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Or performs element-wise logical OR.
func Or(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	if a.dtype != Bool || b.dtype != Bool {
		return nil, errors.New("logical operations require boolean arrays")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)
	vek.Or_Into(data, a.data.([]bool), b.data.([]bool))
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Xor performs element-wise logical XOR.
func Xor(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	if a.dtype != Bool || b.dtype != Bool {
		return nil, errors.New("logical operations require boolean arrays")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)
	vek.Xor_Into(data, a.data.([]bool), b.data.([]bool))
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Not performs element-wise logical NOT.
func (a *NdArray) Not() (*NdArray, error) {
	if a.dtype != Bool {
		return nil, errors.New("logical operations require boolean arrays")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)
	vek.Not_Into(data, a.data.([]bool))
	return &NdArray{shape: a.shape, data: data, dtype: Bool}, nil
}

// Any returns true if any element is true.
func (a *NdArray) Any() (bool, error) {
	if a.dtype != Bool {
		return false, errors.New("logical operations require boolean arrays")
	}
	return vek.Any(a.data.([]bool)), nil
}

// All returns true if all elements are true.
func (a *NdArray) All() (bool, error) {
	if a.dtype != Bool {
		return false, errors.New("logical operations require boolean arrays")
	}
	return vek.All(a.data.([]bool)), nil
}

// None returns true if no elements are true.
func (a *NdArray) None() (bool, error) {
	if a.dtype != Bool {
		return false, errors.New("logical operations require boolean arrays")
	}
	return vek.None(a.data.([]bool)), nil
}

// Count returns the number of true elements.
func (a *NdArray) Count() (int, error) {
	if a.dtype != Bool {
		return 0, errors.New("logical operations require boolean arrays")
	}
	return vek.Count(a.data.([]bool)), nil
}

// Select returns elements from a where mask is true (filtered to 1-D).
func Select(a *NdArray, mask *NdArray) (*NdArray, error) {
	if mask.dtype != Bool {
		return nil, errors.New("mask must be a Bool array")
	}
	if ProdInt(a.shape) != ProdInt(mask.shape) {
		return nil, errors.New("array and mask must have the same number of elements")
	}
	boolData := mask.data.([]bool)
	if a.dtype == Float32 {
		result := vek32.Select(a.data.([]float32), boolData)
		return &NdArray{shape: []int{len(result)}, data: result, dtype: Float32}, nil
	}
	result := vek.Select(a.mustFloat64(), boolData)
	return &NdArray{shape: []int{len(result)}, data: result, dtype: Float64}, nil
}

// --- Utility methods ---

// Copy returns a deep copy of the NdArray.
func (a *NdArray) Copy() *NdArray {
	shapeCopy := make([]int, len(a.shape))
	copy(shapeCopy, a.shape)

	switch a.dtype {
	case Float64:
		src := a.data.([]float64)
		dst := make([]float64, len(src))
		copy(dst, src)
		return &NdArray{shape: shapeCopy, data: dst, dtype: Float64}
	case Float32:
		src := a.data.([]float32)
		dst := make([]float32, len(src))
		copy(dst, src)
		return &NdArray{shape: shapeCopy, data: dst, dtype: Float32}
	default:
		src := a.data.([]bool)
		dst := make([]bool, len(src))
		copy(dst, src)
		return &NdArray{shape: shapeCopy, data: dst, dtype: Bool}
	}
}

// String returns a human-readable representation of the NdArray.
func (a *NdArray) String() string {
	var dtypeStr string
	switch a.dtype {
	case Float64:
		dtypeStr = "float64"
	case Float32:
		dtypeStr = "float32"
	case Bool:
		dtypeStr = "bool"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "NdArray(shape=%v, dtype=%s, data=", a.shape, dtypeStr)

	size := ProdInt(a.shape)
	const maxShow = 10
	switch a.dtype {
	case Float64:
		d := a.data.([]float64)
		b.WriteByte('[')
		for i := range min(size, maxShow) {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%g", d[i])
		}
		if size > maxShow {
			fmt.Fprintf(&b, ", ...(%d more)", size-maxShow)
		}
		b.WriteByte(']')
	case Float32:
		d := a.data.([]float32)
		b.WriteByte('[')
		for i := range min(size, maxShow) {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%g", d[i])
		}
		if size > maxShow {
			fmt.Fprintf(&b, ", ...(%d more)", size-maxShow)
		}
		b.WriteByte(']')
	case Bool:
		d := a.data.([]bool)
		b.WriteByte('[')
		for i := range min(size, maxShow) {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%t", d[i])
		}
		if size > maxShow {
			fmt.Fprintf(&b, ", ...(%d more)", size-maxShow)
		}
		b.WriteByte(']')
	}

	b.WriteByte(')')
	return b.String()
}

func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
