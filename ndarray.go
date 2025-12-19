package ndvek

import (
	"errors"
	"fmt"

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
	Data  any // []float64, []float32, or []bool
	dtype DType
}

func (a *NdArray) DType() DType {
	return a.dtype
}

// NewNdArray creates a new NdArray given a shape and initial data.
// Data can be []float64, []float32, or []bool.
func NewNdArray(shape []int, data any) (*NdArray, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

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

	return &NdArray{shape: shape, Data: data, dtype: dtype}, nil
}

// broadcastShapes finds a broadcasted shape from two shapes.
func broadcastShapes(shape1, shape2 []int) ([]int, error) {
	len1, len2 := len(shape1), len(shape2)
	maxLen := len1
	if len2 > len1 {
		maxLen = len2
	}
	broadcastedShape := make([]int, maxLen)

	for i := 0; i < maxLen; i++ {
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
		if dim1 > dim2 {
			broadcastedShape[maxLen-1-i] = dim1
		} else {
			broadcastedShape[maxLen-1-i] = dim2
		}
	}
	return broadcastedShape, nil
}

// broadcastIndex converts a single index to the correct offset for broadcasting.
func broadcastIndex(shape, broadcastShape []int, index int) (int, error) {
	// Check dimensions match broadcasting rules
	if len(shape) > len(broadcastShape) {
		return 0, errors.New("original shape cannot be larger than broadcast shape")
	}

	// Initialize result index in original array
	originalIndex := 0
	stride := 1

	// Traverse from last dimension to first to handle broadcasting correctly
	for i := len(broadcastShape) - 1; i >= 0; i-- {
		broadcastDim := broadcastShape[i]
		shapeDim := 1 // Assume a broadcasted dimension
		if i-len(broadcastShape)+len(shape) >= 0 {
			shapeDim = shape[i-len(broadcastShape)+len(shape)]
		}

		// Current coordinate along this dimension in the broadcasted array
		currentCoord := index % broadcastDim
		index /= broadcastDim

		// Map the coordinate to the original shape (if broadcasted, use 0)
		if shapeDim != broadcastDim && shapeDim == 1 {
			currentCoord = 0
		}

		// Calculate the corresponding index in the original array
		originalIndex += currentCoord * stride
		stride *= shapeDim
	}

	return originalIndex, nil
}

func (a *NdArray) ApplyHadamardOp(op func(float64) float64) error {
	if a.dtype == Float32 {
		// Promote to Float64
		f32Data := a.Data.([]float32)
		f64Data := make([]float64, len(f32Data))
		for i, v := range f32Data {
			f64Data[i] = op(float64(v))
		}
		a.Data = f64Data
		a.dtype = Float64
		return nil
	}

	dSize := len(a.Data.([]float64))
	d := a.Data.([]float64)
	for i := 0; i < dSize; i++ {
		d[i] = op(d[i])
	}
	return nil
}

// ApplyOp applies an arithmetic operation with broadcasting.
// Note: This always returns a Float64 array because op returns float64.
func ApplyOp(a, b *NdArray, op func(float64, float64) float64) (*NdArray, error) {
	bShape, err := broadcastShapes(a.shape, b.shape)
	if err != nil {
		return nil, err
	}

	sizeOf := 1
	for _, dim := range bShape {
		sizeOf *= dim
	}
	resultData := make([]float64, sizeOf)
	result := &NdArray{shape: bShape, Data: resultData, dtype: Float64}

	aData := a.dataAsFloat64()
	bData := b.dataAsFloat64()

	for i := 0; i < len(resultData); i++ {
		aIndex, err := broadcastIndex(a.shape, bShape, i)
		if err != nil {
			panic(err)
		}
		bIndex, err := broadcastIndex(b.shape, bShape, i)
		if err != nil {
			panic(err)
		}
		resultData[i] = op(aData[aIndex], bData[bIndex])
	}
	return result, nil
}

// Add performs element-wise addition with broadcasting.
func Add(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.Add(a.Data.([]float32), b.Data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.Add(a.dataAsFloat64(), b.dataAsFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0}) // Get works for both types and returns float64
		if a.dtype == Float32 && b.dtype == Float32 {
			// If b was effectively float32 (it is if dtype is float32), we can keep result as float32
			// vek32.AddNumber
			return &NdArray{shape: a.shape, Data: vek32.AddNumber(a.Data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		// If b is Float64, promote a to Float64
		return &NdArray{shape: a.shape, Data: vek.AddNumber(a.dataAsFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: b.shape, Data: vek32.AddNumber(b.Data.([]float32), float32(aVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: b.shape, Data: vek.AddNumber(b.dataAsFloat64(), aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x + y })
}

// Subtract performs element-wise subtraction with broadcasting.
func Subtract(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.Sub(a.Data.([]float32), b.Data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.Sub(a.dataAsFloat64(), b.dataAsFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.SubNumber(a.Data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.SubNumber(a.dataAsFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			diff := vek32.SubNumber(b.Data.([]float32), float32(aVal))
			return &NdArray{shape: b.shape, Data: vek32.MulNumber(diff, -1), dtype: Float32}, nil
		}
		// a - b = -(b - a)
		diff := vek.SubNumber(b.dataAsFloat64(), aVal)
		return &NdArray{shape: b.shape, Data: vek.MulNumber(diff, -1), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x - y })
}

// Multiply performs element-wise multiplication with broadcasting.
func Multiply(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.Mul(a.Data.([]float32), b.Data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.Mul(a.dataAsFloat64(), b.dataAsFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.MulNumber(a.Data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.MulNumber(a.dataAsFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: b.shape, Data: vek32.MulNumber(b.Data.([]float32), float32(aVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: b.shape, Data: vek.MulNumber(b.dataAsFloat64(), aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x * y })
}

// Divide performs element-wise division with broadcasting.
func Divide(a, b *NdArray) (*NdArray, error) {
	if shapesEqual(a.shape, b.shape) {
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.Div(a.Data.([]float32), b.Data.([]float32)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.Div(a.dataAsFloat64(), b.dataAsFloat64()), dtype: Float64}, nil
	}
	if ProdInt(b.shape) == 1 {
		bVal, _ := b.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			return &NdArray{shape: a.shape, Data: vek32.DivNumber(a.Data.([]float32), float32(bVal)), dtype: Float32}, nil
		}
		return &NdArray{shape: a.shape, Data: vek.DivNumber(a.dataAsFloat64(), bVal), dtype: Float64}, nil
	}
	if ProdInt(a.shape) == 1 {
		aVal, _ := a.Get([]int{0})
		if a.dtype == Float32 && b.dtype == Float32 {
			inv := vek32.Inv(b.Data.([]float32))
			return &NdArray{shape: b.shape, Data: vek32.MulNumber(inv, float32(aVal)), dtype: Float32}, nil
		}
		// a / b = a * (1/b)
		inv := vek.Inv(b.dataAsFloat64())
		return &NdArray{shape: b.shape, Data: vek.MulNumber(inv, aVal), dtype: Float64}, nil
	}
	return ApplyOp(a, b, func(x, y float64) float64 { return x / y })
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
	return &NdArray{shape: shape, Data: data}

}

func (a *NdArray) AddScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		newData := vek32.AddNumber(a.Data.([]float32), float32(b))
		return &NdArray{shape: a.shape, Data: newData, dtype: Float32}
	}
	newData := vek.AddNumber(a.dataAsFloat64(), b)
	return &NdArray{shape: a.shape, Data: newData, dtype: Float64}
}

func (a *NdArray) SubScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		newData := vek32.AddNumber(a.Data.([]float32), -float32(b))
		return &NdArray{shape: a.shape, Data: newData, dtype: Float32}
	}
	newData := vek.AddNumber(a.dataAsFloat64(), -b)
	return &NdArray{shape: a.shape, Data: newData, dtype: Float64}
}

func (a *NdArray) MulScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		newData := vek32.MulNumber(a.Data.([]float32), float32(b))
		return &NdArray{shape: a.shape, Data: newData, dtype: Float32}
	}
	newData := vek.MulNumber(a.dataAsFloat64(), b)
	return &NdArray{shape: a.shape, Data: newData, dtype: Float64}
}

func (a *NdArray) DivScalar(b float64) *NdArray {
	if a.dtype == Float32 {
		newData := vek32.DivNumber(a.Data.([]float32), float32(b))
		return &NdArray{shape: a.shape, Data: newData, dtype: Float32}
	}
	newData := vek.DivNumber(a.dataAsFloat64(), b)
	return &NdArray{shape: a.shape, Data: newData, dtype: Float64}
}

func (x *NdArray) InsertAxis(pos int) *NdArray {
	rank := len(x.shape)
	if pos < 0 {
		pos += rank + 1
	}

	result := make([]int, rank+1)

	copy(result[:pos], (x.shape)[:pos])

	// Insert the new value
	result[pos] = 1

	// Copy the remaining elements
	copy(result[pos+1:], x.shape[pos:])

	y, err := NewNdArray(result, x.Data)

	if err != nil {
		panic(err)
	}

	return y
}

func (x *NdArray) Reshape(shape []int) *NdArray {
	if !(ProdInt(shape) == ProdInt(x.shape)) {
		return nil
	}
	x.shape = shape
	return x
}
func (a *NdArray) dataAsFloat64() []float64 {
	if a.dtype == Float64 {
		return a.Data.([]float64)
	}
	// Convert Float32 to Float64
	f32Data := a.Data.([]float32)
	f64Data := make([]float64, len(f32Data))
	for i, v := range f32Data {
		f64Data[i] = float64(v)
	}
	return f64Data
}

func (a *NdArray) dataAsFloat32() []float32 {
	if a.dtype == Float32 {
		return a.Data.([]float32)
	}
	// Convert Float64 to Float32
	f64Data := a.Data.([]float64)
	f32Data := make([]float32, len(f64Data))
	for i, v := range f64Data {
		f32Data[i] = float32(v)
	}
	return f32Data
}

func (a *NdArray) Get(index []int) (float64, error) {
	// Check if the index length matches the number of dimensions
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

	if a.dtype == Float64 {
		return a.Data.([]float64)[offset], nil
	}
	return float64(a.Data.([]float32)[offset]), nil
}

func Linspace(start, stop float64, numPoints int) []float64 {
	if numPoints <= 0 {
		return nil
	}

	step := (stop - start) / float64(numPoints-1)
	result := make([]float64, numPoints)
	for i := 0; i < numPoints; i++ {
		result[i] = start + float64(i)*step
	}
	return result
}

func (a *NdArray) Sum() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Sum(a.Data.([]float32)))
	}
	return vek.Sum(a.dataAsFloat64())
}

func (a *NdArray) Mean() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Mean(a.Data.([]float32)))
	}
	return vek.Mean(a.dataAsFloat64())
}

func (a *NdArray) Min() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Min(a.Data.([]float32)))
	}
	return vek.Min(a.dataAsFloat64())
}

func (a *NdArray) Max() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Max(a.Data.([]float32)))
	}
	return vek.Max(a.dataAsFloat64())
}

func (a *NdArray) Abs() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Abs(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Abs(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Neg() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Neg(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Neg(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Sqrt() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Sqrt(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Sqrt(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Prod() float64 {
	if a.dtype == Float32 {
		return float64(vek32.Prod(a.Data.([]float32)))
	}
	return vek.Prod(a.dataAsFloat64())
}

func (a *NdArray) CumSum() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.CumSum(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.CumSum(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) CumProd() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.CumProd(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.CumProd(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Round() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Round(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Round(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Floor() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Floor(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Floor(a.dataAsFloat64()), dtype: Float64}
}

func (a *NdArray) Ceil() *NdArray {
	if a.dtype == Float32 {
		return &NdArray{shape: a.shape, Data: vek32.Ceil(a.Data.([]float32)), dtype: Float32}
	}
	return &NdArray{shape: a.shape, Data: vek.Ceil(a.dataAsFloat64()), dtype: Float64}
}

// Eq performs element-wise equality comparison.
func Eq(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		// TODO: Implement broadcasting for boolean ops
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Eq_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Eq_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else if a.dtype == Bool && b.dtype == Bool {
		// vek does not have Eq for bools, do manual
		aData := a.Data.([]bool)
		bData := b.Data.([]bool)
		for i := 0; i < size; i++ {
			data[i] = aData[i] == bData[i]
		}
	} else {
		// Mixed types or other combinations - convert to float64 for comparison?
		// Or strictly enforce types? Let's convert to float64 for numbers.
		if a.dtype != Bool && b.dtype != Bool {
			vek.Eq_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Neq performs element-wise non-equality comparison.
func Neq(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Neq_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Neq_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else if a.dtype == Bool && b.dtype == Bool {
		aData := a.Data.([]bool)
		bData := b.Data.([]bool)
		for i := 0; i < size; i++ {
			data[i] = aData[i] != bData[i]
		}
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Neq_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Lt performs element-wise less than comparison.
func Lt(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Lt_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Lt_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Lt_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Lte performs element-wise less than or equal comparison.
func Lte(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Lte_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Lte_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Lte_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Gt performs element-wise greater than comparison.
func Gt(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Gt_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Gt_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Gt_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Gte performs element-wise greater than or equal comparison.
func Gte(a, b *NdArray) (*NdArray, error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, errors.New("broadcasting not yet supported for boolean ops")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)

	if a.dtype == Float64 && b.dtype == Float64 {
		vek.Gte_Into(data, a.Data.([]float64), b.Data.([]float64))
	} else if a.dtype == Float32 && b.dtype == Float32 {
		vek32.Gte_Into(data, a.Data.([]float32), b.Data.([]float32))
	} else {
		if a.dtype != Bool && b.dtype != Bool {
			vek.Gte_Into(data, a.dataAsFloat64(), b.dataAsFloat64())
		} else {
			return nil, errors.New("cannot compare boolean with numeric type")
		}
	}
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

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
	vek.And_Into(data, a.Data.([]bool), b.Data.([]bool))
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
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
	vek.Or_Into(data, a.Data.([]bool), b.Data.([]bool))
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
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
	vek.Xor_Into(data, a.Data.([]bool), b.Data.([]bool))
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}, nil
}

// Not performs element-wise logical NOT.
func (a *NdArray) Not() *NdArray {
	if a.dtype != Bool {
		panic("logical operations require boolean arrays")
	}
	size := ProdInt(a.shape)
	data := make([]bool, size)
	vek.Not_Into(data, a.Data.([]bool))
	return &NdArray{shape: a.shape, Data: data, dtype: Bool}
}

// Any returns true if any element is true.
func (a *NdArray) Any() bool {
	if a.dtype != Bool {
		// Or convert to bool? For now strict.
		panic("logical operations require boolean arrays")
	}
	return vek.Any(a.Data.([]bool))
}

// All returns true if all elements are true.
func (a *NdArray) All() bool {
	if a.dtype != Bool {
		panic("logical operations require boolean arrays")
	}
	return vek.All(a.Data.([]bool))
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
