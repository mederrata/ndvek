package ndvek

import (
	"reflect"
	"testing"
)

func TestNewNdArray(t *testing.T) {
	// Test successful creation
	shape := []int{2, 3}
	data := []float64{1, 2, 3, 4, 5, 6}
	arr, err := NewNdArray(shape, data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(arr.Shape(), shape) {
		t.Errorf("expected shape %v, got %v", shape, arr.Shape())
	}
	if !reflect.DeepEqual(arr.Data, data) {
		t.Errorf("expected data %v, got %v", data, arr.Data)
	}

	// Test shape/data mismatch error
	shape = []int{2, 2}
	data = []float64{1, 2, 3}
	_, err = NewNdArray(shape, data)
	if err == nil {
		t.Error("expected error for mismatched shape and data length, got nil")
	}
}

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		shape1, shape2, expected []int
		expectError              bool
	}{
		{[]int{2, 3}, []int{3}, []int{2, 3}, false},
		{[]int{1, 3}, []int{3}, []int{1, 3}, false},
		{[]int{4, 1, 3}, []int{1, 3}, []int{4, 1, 3}, false},
		{[]int{3, 4, 5}, []int{4, 5}, []int{3, 4, 5}, false},
		{[]int{3, 4, 5}, []int{5}, []int{3, 4, 5}, false},
		{[]int{2, 3}, []int{4, 3}, nil, true}, // incompatible shapes
	}

	for _, tt := range tests {
		result, err := broadcastShapes(tt.shape1, tt.shape2)
		if tt.expectError {
			if err == nil {
				t.Errorf("expected error for shapes %v and %v, got nil", tt.shape1, tt.shape2)
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error for shapes %v and %v: %v", tt.shape1, tt.shape2, err)
			} else if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		}
	}
}

func TestArithmeticOperations(t *testing.T) {
	a, _ := NewNdArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b, _ := NewNdArray([]int{3}, []float64{1, 2, 3})
	expected := []float64{2, 4, 6, 5, 7, 9}

	// Test Add operation
	result, err := Add(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("expected %v, got %v", expected, result.Data)
	}

	// Test Subtract operation
	expected = []float64{0, 0, 0, 3, 3, 3}
	result, err = Subtract(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("expected %v, got %v", expected, result.Data)
	}

	// Test Multiply operation
	expected = []float64{1, 4, 9, 4, 10, 18}
	result, err = Multiply(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("expected %v, got %v", expected, result.Data)
	}

	// Test Divide operation
	expected = []float64{1, 1, 1, 4, 2.5, 2}
	result, err = Divide(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("expected %v, got %v", expected, result.Data)
	}
}

func TestSameShapeArithmetic(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b, _ := NewNdArray([]int{2, 2}, []float64{5, 6, 7, 8})

	// Add
	res, _ := Add(a, b)
	expected := []float64{6, 8, 10, 12}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Add: expected %v, got %v", expected, res.Data)
	}

	// Sub
	res, _ = Subtract(a, b)
	expected = []float64{-4, -4, -4, -4}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Sub: expected %v, got %v", expected, res.Data)
	}

	// Mul
	res, _ = Multiply(a, b)
	expected = []float64{5, 12, 21, 32}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Mul: expected %v, got %v", expected, res.Data)
	}

	// Div
	res, _ = Divide(a, b)
	expected = []float64{1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0, 4.0 / 8.0}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Div: expected %v, got %v", expected, res.Data)
	}
}

func TestScalarArithmetic(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	scalarVal := 2.0
	s, _ := NewNdArray([]int{1}, []float64{scalarVal})

	// Add
	res, _ := Add(a, s)
	expected := []float64{3, 4, 5, 6}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Add scalar: expected %v, got %v", expected, res.Data)
	}

	// Sub (a - s)
	res, _ = Subtract(a, s)
	expected = []float64{-1, 0, 1, 2}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Sub scalar: expected %v, got %v", expected, res.Data)
	}

	// Sub (s - a)
	res, _ = Subtract(s, a)
	expected = []float64{1, 0, -1, -2}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Sub scalar reverse: expected %v, got %v", expected, res.Data)
	}

	// Div (a / s)
	res, _ = Divide(a, s)
	expected = []float64{0.5, 1, 1.5, 2}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Div scalar: expected %v, got %v", expected, res.Data)
	}

	// Div (s / a)
	res, _ = Divide(s, a)
	expected = []float64{2, 1, 2.0 / 3.0, 0.5}
	if !reflect.DeepEqual(res.Data, expected) {
		t.Errorf("Div scalar reverse: expected %v, got %v", expected, res.Data)
	}
}

func TestVekExtensions(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, -2, 3, -4})

	// Sum
	if got := a.Sum(); got != -2 {
		t.Errorf("Sum: expected -2, got %v", got)
	}

	// Mean
	if got := a.Mean(); got != -0.5 {
		t.Errorf("Mean: expected -0.5, got %v", got)
	}

	// Min
	if got := a.Min(); got != -4 {
		t.Errorf("Min: expected -4, got %v", got)
	}

	// Max
	if got := a.Max(); got != 3 {
		t.Errorf("Max: expected 3, got %v", got)
	}

	// Abs
	absRes := a.Abs()
	expected := []float64{1, 2, 3, 4}
	if !reflect.DeepEqual(absRes.Data, expected) {
		t.Errorf("Abs: expected %v, got %v", expected, absRes.Data)
	}

	// Neg
	negRes := a.Neg()
	expected = []float64{-1, 2, -3, 4}
	if !reflect.DeepEqual(negRes.Data, expected) {
		t.Errorf("Neg: expected %v, got %v", expected, negRes.Data)
	}

	// Sqrt (using positive data)
	b, _ := NewNdArray([]int{2, 2}, []float64{1, 4, 9, 16})
	sqrtRes := b.Sqrt()
	expected = []float64{1, 2, 3, 4}
	if !reflect.DeepEqual(sqrtRes.Data, expected) {
		t.Errorf("Sqrt: expected %v, got %v", expected, sqrtRes.Data)
	}

	// DivScalar
	divRes := b.DivScalar(2.0)
	expected = []float64{0.5, 2, 4.5, 8}
	if !reflect.DeepEqual(divRes.Data, expected) {
		t.Errorf("DivScalar: expected %v, got %v", expected, divRes.Data)
	}

	// Prod
	if got := a.Prod(); got != 24 {
		t.Errorf("Prod: expected 24, got %v", got)
	}

	// CumSum
	cumSumRes := a.CumSum()
	expected = []float64{1, -1, 2, -2}
	if !reflect.DeepEqual(cumSumRes.Data, expected) {
		t.Errorf("CumSum: expected %v, got %v", expected, cumSumRes.Data)
	}

	// CumProd
	cumProdRes := a.CumProd()
	expected = []float64{1, -2, -6, 24}
	if !reflect.DeepEqual(cumProdRes.Data, expected) {
		t.Errorf("CumProd: expected %v, got %v", expected, cumProdRes.Data)
	}

	// Round, Floor, Ceil
	c, _ := NewNdArray([]int{3}, []float64{1.1, 1.9, -1.1})

	roundRes := c.Round()
	expected = []float64{1, 2, -1}
	if !reflect.DeepEqual(roundRes.Data, expected) {
		t.Errorf("Round: expected %v, got %v", expected, roundRes.Data)
	}

	floorRes := c.Floor()
	expected = []float64{1, 1, -2}
	if !reflect.DeepEqual(floorRes.Data, expected) {
		t.Errorf("Floor: expected %v, got %v", expected, floorRes.Data)
	}

	ceilRes := c.Ceil()
	expected = []float64{2, 2, -1}
	if !reflect.DeepEqual(ceilRes.Data, expected) {
		t.Errorf("Ceil: expected %v, got %v", expected, ceilRes.Data)
	}
}
