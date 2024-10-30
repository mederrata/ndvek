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
	if !reflect.DeepEqual(arr.data, data) {
		t.Errorf("expected data %v, got %v", data, arr.data)
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
	if !reflect.DeepEqual(result.data, expected) {
		t.Errorf("expected %v, got %v", expected, result.data)
	}

	// Test Subtract operation
	expected = []float64{0, 0, 0, 3, 3, 3}
	result, err = Subtract(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.data, expected) {
		t.Errorf("expected %v, got %v", expected, result.data)
	}

	// Test Multiply operation
	expected = []float64{1, 4, 9, 4, 10, 18}
	result, err = Multiply(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.data, expected) {
		t.Errorf("expected %v, got %v", expected, result.data)
	}

	// Test Divide operation
	expected = []float64{1, 1, 1, 4, 2.5, 2}
	result, err = Divide(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.data, expected) {
		t.Errorf("expected %v, got %v", expected, result.data)
	}
}
