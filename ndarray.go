package ndvek

import (
	"errors"
	"fmt"
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

// Shape returns the shape of the ndarray.
func (a *NdArray) Shape() []int {
	return a.shape
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
func broadcastIndex(shape, broadcastShape []int, index int) int {
	offset := 0
	factor := 1
	for i := len(shape) - 1; i >= 0; i-- {
		dim := shape[i]
		bdim := broadcastShape[i+len(broadcastShape)-len(shape)]
		if dim != bdim && dim == 1 {
			offset += (index / factor) % bdim * factor
		} else {
			offset += (index / factor % dim) * factor
		}
		factor *= bdim
	}
	return offset
}

// applyOp applies an arithmetic operation with broadcasting.
func applyOp(a, b *NdArray, op func(float64, float64) float64) (*NdArray, error) {
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
		aIndex := broadcastIndex(a.shape, bShape, i)
		bIndex := broadcastIndex(b.shape, bShape, i)
		result.data[i] = op(a.data[aIndex], b.data[bIndex])
	}
	return result, nil
}

// Add performs element-wise addition with broadcasting.
func Add(a, b *NdArray) (*NdArray, error) {
	return applyOp(a, b, func(x, y float64) float64 { return x + y })
}

// Subtract performs element-wise subtraction with broadcasting.
func Subtract(a, b *NdArray) (*NdArray, error) {
	return applyOp(a, b, func(x, y float64) float64 { return x - y })
}

// Multiply performs element-wise multiplication with broadcasting.
func Multiply(a, b *NdArray) (*NdArray, error) {
	return applyOp(a, b, func(x, y float64) float64 { return x * y })
}

// Divide performs element-wise division with broadcasting.
func Divide(a, b *NdArray) (*NdArray, error) {
	return applyOp(a, b, func(x, y float64) float64 { return x / y })
}
