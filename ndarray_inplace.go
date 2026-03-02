package ndvek

import (
	"errors"

	"github.com/viterin/vek"
	"github.com/viterin/vek/vek32"
)

// AddInPlace performs element-wise addition: a += b.
func (a *NdArray) AddInPlace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Add_Inplace(a.data.([]float32), b.data.([]float32))
		return nil
	}
	vek.Add_Inplace(a.data.([]float64), b.mustFloat64())
	return nil
}

// SubtractInPlace performs element-wise subtraction: a -= b.
func (a *NdArray) SubtractInPlace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Sub_Inplace(a.data.([]float32), b.data.([]float32))
		return nil
	}
	vek.Sub_Inplace(a.data.([]float64), b.mustFloat64())
	return nil
}

// MultiplyInPlace performs element-wise multiplication: a *= b.
func (a *NdArray) MultiplyInPlace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Mul_Inplace(a.data.([]float32), b.data.([]float32))
		return nil
	}
	vek.Mul_Inplace(a.data.([]float64), b.mustFloat64())
	return nil
}

// DivideInPlace performs element-wise division: a /= b.
func (a *NdArray) DivideInPlace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Div_Inplace(a.data.([]float32), b.data.([]float32))
		return nil
	}
	vek.Div_Inplace(a.data.([]float64), b.mustFloat64())
	return nil
}

// AddScalarInPlace adds a scalar to each element: a += b.
func (a *NdArray) AddScalarInPlace(b float64) {
	if a.dtype == Float32 {
		vek32.AddNumber_Inplace(a.data.([]float32), float32(b))
	} else {
		vek.AddNumber_Inplace(a.data.([]float64), b)
	}
}

// SubScalarInPlace subtracts a scalar from each element: a -= b.
func (a *NdArray) SubScalarInPlace(b float64) {
	if a.dtype == Float32 {
		vek32.SubNumber_Inplace(a.data.([]float32), float32(b))
	} else {
		vek.SubNumber_Inplace(a.data.([]float64), b)
	}
}

// MulScalarInPlace multiplies each element by a scalar: a *= b.
func (a *NdArray) MulScalarInPlace(b float64) {
	if a.dtype == Float32 {
		vek32.MulNumber_Inplace(a.data.([]float32), float32(b))
	} else {
		vek.MulNumber_Inplace(a.data.([]float64), b)
	}
}

// DivScalarInPlace divides each element by a scalar: a /= b.
func (a *NdArray) DivScalarInPlace(b float64) {
	if a.dtype == Float32 {
		vek32.DivNumber_Inplace(a.data.([]float32), float32(b))
	} else {
		vek.DivNumber_Inplace(a.data.([]float64), b)
	}
}

// AbsInPlace computes the absolute value in-place.
func (a *NdArray) AbsInPlace() {
	if a.dtype == Float32 {
		vek32.Abs_Inplace(a.data.([]float32))
	} else {
		vek.Abs_Inplace(a.data.([]float64))
	}
}

// NegInPlace computes the negation in-place.
func (a *NdArray) NegInPlace() {
	if a.dtype == Float32 {
		vek32.Neg_Inplace(a.data.([]float32))
	} else {
		vek.Neg_Inplace(a.data.([]float64))
	}
}

// SqrtInPlace computes the square root in-place.
func (a *NdArray) SqrtInPlace() {
	if a.dtype == Float32 {
		vek32.Sqrt_Inplace(a.data.([]float32))
	} else {
		vek.Sqrt_Inplace(a.data.([]float64))
	}
}

// RoundInPlace rounds elements in-place.
func (a *NdArray) RoundInPlace() {
	if a.dtype == Float32 {
		vek32.Round_Inplace(a.data.([]float32))
	} else {
		vek.Round_Inplace(a.data.([]float64))
	}
}

// FloorInPlace floors elements in-place.
func (a *NdArray) FloorInPlace() {
	if a.dtype == Float32 {
		vek32.Floor_Inplace(a.data.([]float32))
	} else {
		vek.Floor_Inplace(a.data.([]float64))
	}
}

// CeilInPlace ceils elements in-place.
func (a *NdArray) CeilInPlace() {
	if a.dtype == Float32 {
		vek32.Ceil_Inplace(a.data.([]float32))
	} else {
		vek.Ceil_Inplace(a.data.([]float64))
	}
}

// CumSumInPlace computes cumulative sum in-place.
func (a *NdArray) CumSumInPlace() {
	if a.dtype == Float32 {
		vek32.CumSum_Inplace(a.data.([]float32))
	} else {
		vek.CumSum_Inplace(a.data.([]float64))
	}
}

// CumProdInPlace computes cumulative product in-place.
func (a *NdArray) CumProdInPlace() {
	if a.dtype == Float32 {
		vek32.CumProd_Inplace(a.data.([]float32))
	} else {
		vek.CumProd_Inplace(a.data.([]float64))
	}
}
