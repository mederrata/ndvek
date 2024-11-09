package ndvek

import (
	"errors"
	"fmt"

	"github.com/viterin/vek"
)

// NdArray represents a multi-dimensional array with shape and data.
type NdArray struct {
	shape []int
	data  []float64
}

// NewNdArray creates a new NdArray given a shape and initial data.
func NewNdArray(shape []int, data []float64) (*NdArray, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != len(data) {
		return nil, errors.New("data length does not match shape dimensions")
	}
	return &NdArray{shape: shape, data: data}, nil
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
	dSize := len(a.data)
	for i := 0; i < dSize; i++ {
		a.data[i] = op(a.data[i])
	}
	return nil
}

// applyOp applies an arithmetic operation with broadcasting.
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

	result := &NdArray{shape: bShape, data: resultData}

	for i := 0; i < len(result.data); i++ {
		aIndex, err := broadcastIndex(a.shape, bShape, i)
		if err != nil {
			panic(err)
		}
		bIndex, err := broadcastIndex(b.shape, bShape, i)
		if err != nil {
			panic(err)
		}
		result.data[i] = op(a.data[aIndex], b.data[bIndex])
	}
	return result, nil
}

// Add performs element-wise addition with broadcasting.
func Add(a, b *NdArray) (*NdArray, error) {
	return ApplyOp(a, b, func(x, y float64) float64 { return x + y })
}

// Subtract performs element-wise subtraction with broadcasting.
func Subtract(a, b *NdArray) (*NdArray, error) {
	return ApplyOp(a, b, func(x, y float64) float64 { return x - y })
}

// Multiply performs element-wise multiplication with broadcasting.
func Multiply(a, b *NdArray) (*NdArray, error) {
	return ApplyOp(a, b, func(x, y float64) float64 { return x * y })
}

// Divide performs element-wise division with broadcasting.
func Divide(a, b *NdArray) (*NdArray, error) {
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
	for i := 0; i < size; i++ {
		data[i] = 0
	}
	return &NdArray{shape: shape, data: data}

}

func (a *NdArray) AddScalar(b float64) *NdArray {
	newData := vek.AddNumber(a.data, b)
	newVek, err := NewNdArray(a.shape, newData)
	if err != nil {
		panic(err)
	}
	return newVek
}

func (a *NdArray) SubScalar(b float64) *NdArray {
	newData := vek.AddNumber(a.data, -b)
	newVek, err := NewNdArray(a.shape, newData)
	if err != nil {
		panic(err)
	}
	return newVek
}

func (a *NdArray) MulScalar(b float64) *NdArray {
	newData := vek.MulNumber(a.data, b)
	newVek, err := NewNdArray(a.shape, newData)
	if err != nil {
		panic(err)
	}
	return newVek
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

	y, err := NewNdArray(result, x.data)

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
func (a *NdArray) Get(index []int) (float64, error) {
	// Check if the index length matches the number of dimensions
	if len(index) != len(a.shape) {
		return 0, errors.New("index length does not match array dimensions")
	}

	// Validate and calculate offset in row-major order
	offset := 0
	for i, coord := range index {
		dim := a.shape[i]

		// Check for out-of-bounds access
		if coord >= dim || coord < 0 {
			return 0, errors.New("index out of bounds")
		}

		// Calculate offset based on row and column indices (assuming C-style indexing)
		if i == 0 { // First dimension (rows)
			offset = coord * a.shape[1] // Multiply by number of elements in a row
		} else {
			offset += coord // Add column index for subsequent dimensions
		}
	}

	return a.data[offset], nil
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
