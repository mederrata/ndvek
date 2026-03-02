package ndvek

import (
	"math"
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
	if !reflect.DeepEqual(arr.Float64Data(), data) {
		t.Errorf("expected data %v, got %v", data, arr.Float64Data())
	}

	// Test shape/data mismatch error
	shape = []int{2, 2}
	data = []float64{1, 2, 3}
	_, err = NewNdArray(shape, data)
	if err == nil {
		t.Error("expected error for mismatched shape and data length, got nil")
	}

	// Test shape aliasing protection
	origShape := []int{2, 3}
	arr2, _ := NewNdArray(origShape, []float64{1, 2, 3, 4, 5, 6})
	origShape[0] = 99
	if arr2.Shape()[0] == 99 {
		t.Error("NewNdArray should copy the shape slice to prevent aliasing")
	}

	// Test typed accessors
	f32Arr, _ := NewNdArray([]int{2}, []float32{1.0, 2.0})
	if f32Arr.Float64Data() != nil {
		t.Error("Float64Data should return nil for Float32 array")
	}
	if f32Arr.Float32Data() == nil {
		t.Error("Float32Data should return non-nil for Float32 array")
	}
	if f32Arr.BoolData() != nil {
		t.Error("BoolData should return nil for Float32 array")
	}

	boolArr, _ := NewNdArray([]int{2}, []bool{true, false})
	if boolArr.BoolData() == nil {
		t.Error("BoolData should return non-nil for Bool array")
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
		{[]int{2, 3}, []int{4, 3}, nil, true},
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

	result, err := Add(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Float64Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Float64Data())
	}

	expected = []float64{0, 0, 0, 3, 3, 3}
	result, err = Subtract(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Float64Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Float64Data())
	}

	expected = []float64{1, 4, 9, 4, 10, 18}
	result, err = Multiply(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Float64Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Float64Data())
	}

	expected = []float64{1, 1, 1, 4, 2.5, 2}
	result, err = Divide(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(result.Float64Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Float64Data())
	}
}

func TestSameShapeArithmetic(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b, _ := NewNdArray([]int{2, 2}, []float64{5, 6, 7, 8})

	res, _ := Add(a, b)
	expected := []float64{6, 8, 10, 12}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Add: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Subtract(a, b)
	expected = []float64{-4, -4, -4, -4}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Sub: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Multiply(a, b)
	expected = []float64{5, 12, 21, 32}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Mul: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Divide(a, b)
	expected = []float64{1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0, 4.0 / 8.0}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Div: expected %v, got %v", expected, res.Float64Data())
	}
}

func TestScalarArithmetic(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	scalarVal := 2.0
	s, _ := NewNdArray([]int{1}, []float64{scalarVal})

	res, _ := Add(a, s)
	expected := []float64{3, 4, 5, 6}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Add scalar: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Subtract(a, s)
	expected = []float64{-1, 0, 1, 2}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Sub scalar: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Subtract(s, a)
	expected = []float64{1, 0, -1, -2}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Sub scalar reverse: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Divide(a, s)
	expected = []float64{0.5, 1, 1.5, 2}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Div scalar: expected %v, got %v", expected, res.Float64Data())
	}

	res, _ = Divide(s, a)
	expected = []float64{2, 1, 2.0 / 3.0, 0.5}
	if !reflect.DeepEqual(res.Float64Data(), expected) {
		t.Errorf("Div scalar reverse: expected %v, got %v", expected, res.Float64Data())
	}
}

func TestVekExtensions(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, -2, 3, -4})

	if got := a.Sum(); got != -2 {
		t.Errorf("Sum: expected -2, got %v", got)
	}

	if got := a.Mean(); got != -0.5 {
		t.Errorf("Mean: expected -0.5, got %v", got)
	}

	if got := a.Min(); got != -4 {
		t.Errorf("Min: expected -4, got %v", got)
	}

	if got := a.Max(); got != 3 {
		t.Errorf("Max: expected 3, got %v", got)
	}

	absRes := a.Abs()
	expected := []float64{1, 2, 3, 4}
	if !reflect.DeepEqual(absRes.Float64Data(), expected) {
		t.Errorf("Abs: expected %v, got %v", expected, absRes.Float64Data())
	}

	negRes := a.Neg()
	expected = []float64{-1, 2, -3, 4}
	if !reflect.DeepEqual(negRes.Float64Data(), expected) {
		t.Errorf("Neg: expected %v, got %v", expected, negRes.Float64Data())
	}

	b, _ := NewNdArray([]int{2, 2}, []float64{1, 4, 9, 16})
	sqrtRes := b.Sqrt()
	expected = []float64{1, 2, 3, 4}
	if !reflect.DeepEqual(sqrtRes.Float64Data(), expected) {
		t.Errorf("Sqrt: expected %v, got %v", expected, sqrtRes.Float64Data())
	}

	divRes := b.DivScalar(2.0)
	expected = []float64{0.5, 2, 4.5, 8}
	if !reflect.DeepEqual(divRes.Float64Data(), expected) {
		t.Errorf("DivScalar: expected %v, got %v", expected, divRes.Float64Data())
	}

	if got := a.Prod(); got != 24 {
		t.Errorf("Prod: expected 24, got %v", got)
	}

	cumSumRes := a.CumSum()
	expected = []float64{1, -1, 2, -2}
	if !reflect.DeepEqual(cumSumRes.Float64Data(), expected) {
		t.Errorf("CumSum: expected %v, got %v", expected, cumSumRes.Float64Data())
	}

	cumProdRes := a.CumProd()
	expected = []float64{1, -2, -6, 24}
	if !reflect.DeepEqual(cumProdRes.Float64Data(), expected) {
		t.Errorf("CumProd: expected %v, got %v", expected, cumProdRes.Float64Data())
	}

	c, _ := NewNdArray([]int{3}, []float64{1.1, 1.9, -1.1})

	roundRes := c.Round()
	expected = []float64{1, 2, -1}
	if !reflect.DeepEqual(roundRes.Float64Data(), expected) {
		t.Errorf("Round: expected %v, got %v", expected, roundRes.Float64Data())
	}

	floorRes := c.Floor()
	expected = []float64{1, 1, -2}
	if !reflect.DeepEqual(floorRes.Float64Data(), expected) {
		t.Errorf("Floor: expected %v, got %v", expected, floorRes.Float64Data())
	}

	ceilRes := c.Ceil()
	expected = []float64{2, 2, -1}
	if !reflect.DeepEqual(ceilRes.Float64Data(), expected) {
		t.Errorf("Ceil: expected %v, got %v", expected, ceilRes.Float64Data())
	}
}

func TestBooleanOperations(t *testing.T) {
	boolData := []bool{true, false, true, false}
	a, _ := NewNdArray([]int{4}, boolData)
	if a.DType() != Bool {
		t.Errorf("Expected Bool dtype")
	}

	notA, err := a.Not()
	if err != nil {
		t.Fatalf("Not: unexpected error: %v", err)
	}
	expectedNot := []bool{false, true, false, true}
	if !reflect.DeepEqual(notA.BoolData(), expectedNot) {
		t.Errorf("Not: expected %v, got %v", expectedNot, notA.BoolData())
	}

	bData := []bool{true, true, false, false}
	b, _ := NewNdArray([]int{4}, bData)
	andRes, _ := And(a, b)
	expectedAnd := []bool{true, false, false, false}
	if !reflect.DeepEqual(andRes.BoolData(), expectedAnd) {
		t.Errorf("And: expected %v, got %v", expectedAnd, andRes.BoolData())
	}

	orRes, _ := Or(a, b)
	expectedOr := []bool{true, true, true, false}
	if !reflect.DeepEqual(orRes.BoolData(), expectedOr) {
		t.Errorf("Or: expected %v, got %v", expectedOr, orRes.BoolData())
	}

	xorRes, _ := Xor(a, b)
	expectedXor := []bool{false, true, true, false}
	if !reflect.DeepEqual(xorRes.BoolData(), expectedXor) {
		t.Errorf("Xor: expected %v, got %v", expectedXor, xorRes.BoolData())
	}

	anyResult, _ := a.Any()
	if !anyResult {
		t.Error("Any: expected true")
	}
	allResult, _ := a.All()
	if allResult {
		t.Error("All: expected false")
	}
	c, _ := NewNdArray([]int{2}, []bool{true, true})
	allC, _ := c.All()
	if !allC {
		t.Error("All: expected true")
	}

	noneRes, _ := a.None()
	if noneRes {
		t.Error("None: expected false for array with true elements")
	}
	d, _ := NewNdArray([]int{2}, []bool{false, false})
	noneD, _ := d.None()
	if !noneD {
		t.Error("None: expected true for all-false array")
	}

	countRes, _ := a.Count()
	if countRes != 2 {
		t.Errorf("Count: expected 2, got %d", countRes)
	}

	// Comparisons
	f1, _ := NewNdArray([]int{4}, []float64{1, 2, 3, 4})
	f2, _ := NewNdArray([]int{4}, []float64{1, 2, 4, 3})

	eqRes, _ := Eq(f1, f2)
	expectedEq := []bool{true, true, false, false}
	if !reflect.DeepEqual(eqRes.BoolData(), expectedEq) {
		t.Errorf("Eq: expected %v, got %v", expectedEq, eqRes.BoolData())
	}

	ltRes, _ := Lt(f1, f2)
	expectedLt := []bool{false, false, true, false}
	if !reflect.DeepEqual(ltRes.BoolData(), expectedLt) {
		t.Errorf("Lt: expected %v, got %v", expectedLt, ltRes.BoolData())
	}

	gtRes, _ := Gt(f1, f2)
	expectedGt := []bool{false, false, false, true}
	if !reflect.DeepEqual(gtRes.BoolData(), expectedGt) {
		t.Errorf("Gt: expected %v, got %v", expectedGt, gtRes.BoolData())
	}
}

func TestReshape(t *testing.T) {
	a, _ := NewNdArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})

	reshaped, err := a.Reshape([]int{3, 2})
	if err != nil {
		t.Fatalf("Reshape: unexpected error: %v", err)
	}
	if !reflect.DeepEqual(reshaped.Shape(), []int{3, 2}) {
		t.Errorf("Reshape: expected shape [3 2], got %v", reshaped.Shape())
	}

	_, err = a.Reshape([]int{5, 5})
	if err == nil {
		t.Error("Reshape: expected error for incompatible shape, got nil")
	}
}

func TestInsertAxis(t *testing.T) {
	a, _ := NewNdArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})

	expanded, err := a.InsertAxis(0)
	if err != nil {
		t.Fatalf("InsertAxis: unexpected error: %v", err)
	}
	if !reflect.DeepEqual(expanded.Shape(), []int{1, 2, 3}) {
		t.Errorf("InsertAxis(0): expected shape [1 2 3], got %v", expanded.Shape())
	}

	expanded2, err := a.InsertAxis(-1)
	if err != nil {
		t.Fatalf("InsertAxis: unexpected error: %v", err)
	}
	if !reflect.DeepEqual(expanded2.Shape(), []int{2, 3, 1}) {
		t.Errorf("InsertAxis(-1): expected shape [2 3 1], got %v", expanded2.Shape())
	}
}

func TestCopy(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b := a.Copy()

	// Modify original
	a.Float64Data()[0] = 99

	if b.Float64Data()[0] == 99 {
		t.Error("Copy: modifying original should not affect the copy")
	}
}

func TestString(t *testing.T) {
	a, _ := NewNdArray([]int{2, 2}, []float64{1, 2, 3, 4})
	s := a.String()
	if s == "" {
		t.Error("String: expected non-empty string")
	}
}

func TestLinspace(t *testing.T) {
	// Normal case
	result := Linspace(0, 1, 5)
	if len(result) != 5 {
		t.Fatalf("Linspace: expected 5 points, got %d", len(result))
	}
	if result[0] != 0 || result[4] != 1 {
		t.Errorf("Linspace: expected [0, ..., 1], got %v", result)
	}

	// Single point case (was a div-by-zero bug)
	single := Linspace(5, 10, 1)
	if len(single) != 1 || single[0] != 5 {
		t.Errorf("Linspace(5,10,1): expected [5], got %v", single)
	}

	// Zero points
	empty := Linspace(0, 1, 0)
	if empty != nil {
		t.Errorf("Linspace(0,1,0): expected nil, got %v", empty)
	}
}

func TestZerosAndOnes(t *testing.T) {
	z := Zeros([]int{3, 2})
	if z.DType() != Float64 {
		t.Errorf("Zeros: expected Float64 dtype, got %v", z.DType())
	}
	if z.Sum() != 0 {
		t.Errorf("Zeros: expected sum 0, got %v", z.Sum())
	}

	o := Ones([]int{3, 2})
	if o.DType() != Float64 {
		t.Errorf("Ones: expected Float64 dtype, got %v", o.DType())
	}
	if o.Sum() != 6 {
		t.Errorf("Ones: expected sum 6, got %v", o.Sum())
	}
}

func TestNewVekOperations(t *testing.T) {
	a, _ := NewNdArray([]int{4}, []float64{1, 2, 3, 4})
	b, _ := NewNdArray([]int{4}, []float64{2, 3, 1, 2})

	// Pow
	powRes, err := Pow(a, b)
	if err != nil {
		t.Fatalf("Pow: unexpected error: %v", err)
	}
	expectedPow := []float64{1, 8, 3, 16}
	if !reflect.DeepEqual(powRes.Float64Data(), expectedPow) {
		t.Errorf("Pow: expected %v, got %v", expectedPow, powRes.Float64Data())
	}

	// Minimum / Maximum
	minRes, _ := Minimum(a, b)
	expectedMin := []float64{1, 2, 1, 2}
	if !reflect.DeepEqual(minRes.Float64Data(), expectedMin) {
		t.Errorf("Minimum: expected %v, got %v", expectedMin, minRes.Float64Data())
	}
	maxRes, _ := Maximum(a, b)
	expectedMax := []float64{2, 3, 3, 4}
	if !reflect.DeepEqual(maxRes.Float64Data(), expectedMax) {
		t.Errorf("Maximum: expected %v, got %v", expectedMax, maxRes.Float64Data())
	}

	// Inv
	invRes := a.Inv()
	expectedInv := []float64{1, 0.5, 1.0 / 3.0, 0.25}
	if !reflect.DeepEqual(invRes.Float64Data(), expectedInv) {
		t.Errorf("Inv: expected %v, got %v", expectedInv, invRes.Float64Data())
	}

	// Median
	c, _ := NewNdArray([]int{5}, []float64{3, 1, 4, 1, 5})
	med := c.Median()
	if med != 3 {
		t.Errorf("Median: expected 3, got %v", med)
	}

	// ArgMin / ArgMax
	if a.ArgMin() != 0 {
		t.Errorf("ArgMin: expected 0, got %d", a.ArgMin())
	}
	if a.ArgMax() != 3 {
		t.Errorf("ArgMax: expected 3, got %d", a.ArgMax())
	}

	// Dot
	dotRes, err := Dot(a, b)
	if err != nil {
		t.Fatalf("Dot: unexpected error: %v", err)
	}
	// 1*2 + 2*3 + 3*1 + 4*2 = 2+6+3+8 = 19
	if dotRes != 19 {
		t.Errorf("Dot: expected 19, got %v", dotRes)
	}

	// Norm
	d, _ := NewNdArray([]int{2}, []float64{3, 4})
	if d.Norm() != 5 {
		t.Errorf("Norm: expected 5, got %v", d.Norm())
	}

	// ManhattanNorm
	e, _ := NewNdArray([]int{3}, []float64{1, -2, 3})
	if e.ManhattanNorm() != 6 {
		t.Errorf("ManhattanNorm: expected 6, got %v", e.ManhattanNorm())
	}

	// Distance
	f1, _ := NewNdArray([]int{2}, []float64{0, 0})
	f2, _ := NewNdArray([]int{2}, []float64{3, 4})
	dist, _ := Distance(f1, f2)
	if dist != 5 {
		t.Errorf("Distance: expected 5, got %v", dist)
	}

	// CosineSimilarity
	g1, _ := NewNdArray([]int{3}, []float64{1, 0, 0})
	g2, _ := NewNdArray([]int{3}, []float64{1, 0, 0})
	cs, _ := CosineSimilarity(g1, g2)
	if math.Abs(cs-1.0) > 1e-10 {
		t.Errorf("CosineSimilarity: expected 1.0, got %v", cs)
	}
}

func TestTranscendentals(t *testing.T) {
	a, _ := NewNdArray([]int{3}, []float64{0, math.Pi / 2, math.Pi})
	const eps = 1e-10

	sinRes := a.Sin()
	sinData := sinRes.Float64Data()
	if math.Abs(sinData[0]-0) > eps || math.Abs(sinData[1]-1) > eps || math.Abs(sinData[2]-0) > eps {
		t.Errorf("Sin: unexpected result %v", sinData)
	}

	cosRes := a.Cos()
	cosData := cosRes.Float64Data()
	if math.Abs(cosData[0]-1) > eps || math.Abs(cosData[1]-0) > eps || math.Abs(cosData[2]-(-1)) > eps {
		t.Errorf("Cos: unexpected result %v", cosData)
	}

	b, _ := NewNdArray([]int{3}, []float64{0, 1, 2})
	expRes := b.Exp()
	expData := expRes.Float64Data()
	if math.Abs(expData[0]-1) > eps || math.Abs(expData[1]-math.E) > eps {
		t.Errorf("Exp: unexpected result %v", expData)
	}

	c, _ := NewNdArray([]int{3}, []float64{1, math.E, 100})
	logRes := c.Log()
	logData := logRes.Float64Data()
	if math.Abs(logData[0]-0) > eps || math.Abs(logData[1]-1) > eps {
		t.Errorf("Log: unexpected result %v", logData)
	}

	d, _ := NewNdArray([]int{3}, []float64{1, 2, 8})
	log2Res := d.Log2()
	log2Data := log2Res.Float64Data()
	if math.Abs(log2Data[0]-0) > eps || math.Abs(log2Data[1]-1) > eps || math.Abs(log2Data[2]-3) > eps {
		t.Errorf("Log2: unexpected result %v", log2Data)
	}

	e, _ := NewNdArray([]int{3}, []float64{1, 10, 100})
	log10Res := e.Log10()
	log10Data := log10Res.Float64Data()
	if math.Abs(log10Data[0]-0) > eps || math.Abs(log10Data[1]-1) > eps || math.Abs(log10Data[2]-2) > eps {
		t.Errorf("Log10: unexpected result %v", log10Data)
	}
}

func TestSelect(t *testing.T) {
	mask, _ := NewNdArray([]int{4}, []bool{true, false, true, false})
	a, _ := NewNdArray([]int{4}, []float64{1, 2, 3, 4})

	result, err := Select(a, mask)
	if err != nil {
		t.Fatalf("Select: unexpected error: %v", err)
	}
	expected := []float64{1, 3}
	if !reflect.DeepEqual(result.Float64Data(), expected) {
		t.Errorf("Select: expected %v, got %v", expected, result.Float64Data())
	}
	if !reflect.DeepEqual(result.Shape(), []int{2}) {
		t.Errorf("Select: expected shape [2], got %v", result.Shape())
	}
}

func TestInPlaceOperations(t *testing.T) {
	a, _ := NewNdArray([]int{4}, []float64{1, 2, 3, 4})
	b, _ := NewNdArray([]int{4}, []float64{5, 6, 7, 8})

	err := a.AddInPlace(b)
	if err != nil {
		t.Fatalf("AddInPlace: unexpected error: %v", err)
	}
	expected := []float64{6, 8, 10, 12}
	if !reflect.DeepEqual(a.Float64Data(), expected) {
		t.Errorf("AddInPlace: expected %v, got %v", expected, a.Float64Data())
	}
}

func TestApplyOpErrorHandling(t *testing.T) {
	// Bool arrays should return error through ApplyOp, not panic
	boolArr, _ := NewNdArray([]int{2}, []bool{true, false})
	f64Arr, _ := NewNdArray([]int{3}, []float64{1, 2, 3})

	_, err := ApplyOp(boolArr, f64Arr, func(x, y float64) float64 { return x + y })
	if err == nil {
		t.Error("ApplyOp with Bool array should return error")
	}
}

func TestErrorConsistency(t *testing.T) {
	// Not on non-bool should return error, not panic
	f64Arr, _ := NewNdArray([]int{2}, []float64{1, 2})
	_, err := f64Arr.Not()
	if err == nil {
		t.Error("Not on non-bool should return error")
	}

	_, err = f64Arr.Any()
	if err == nil {
		t.Error("Any on non-bool should return error")
	}

	_, err = f64Arr.All()
	if err == nil {
		t.Error("All on non-bool should return error")
	}
}
