package ndvek

import (
	"errors"

	"github.com/viterin/vek"
	"github.com/viterin/vek/vek32"
)

// Add_Inplace performs element-wise addition: a += b.
func (a *NdArray) Add_Inplace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Add_Inplace(a.Data.([]float32), b.Data.([]float32))
		return nil
	}
	vek.Add_Inplace(a.Data.([]float64), b.dataAsFloat64())
	return nil
}

// Subtract_Inplace performs element-wise subtraction: a -= b.
func (a *NdArray) Subtract_Inplace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Sub_Inplace(a.Data.([]float32), b.Data.([]float32))
		return nil
	}
	vek.Sub_Inplace(a.Data.([]float64), b.dataAsFloat64())
	return nil
}

// Multiply_Inplace performs element-wise multiplication: a *= b.
func (a *NdArray) Multiply_Inplace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Mul_Inplace(a.Data.([]float32), b.Data.([]float32))
		return nil
	}
	vek.Mul_Inplace(a.Data.([]float64), b.dataAsFloat64())
	return nil
}

// Divide_Inplace performs element-wise division: a /= b.
func (a *NdArray) Divide_Inplace(b *NdArray) error {
	if !shapesEqual(a.shape, b.shape) {
		return errors.New("shapes must be equal for in-place operation")
	}
	if a.dtype == Float32 {
		if b.dtype == Float64 {
			return errors.New("cannot operate on Float32 with Float64 in-place")
		}
		vek32.Div_Inplace(a.Data.([]float32), b.Data.([]float32))
		return nil
	}
	vek.Div_Inplace(a.Data.([]float64), b.dataAsFloat64())
	return nil
}

// AddScalar_Inplace adds a scalar to each element: a += b.
func (a *NdArray) AddScalar_Inplace(b float64) {
	if a.dtype == Float32 {
		vek32.AddNumber_Inplace(a.Data.([]float32), float32(b))
	} else {
		vek.AddNumber_Inplace(a.Data.([]float64), b)
	}
}

// SubScalar_Inplace subtracts a scalar from each element: a -= b.
func (a *NdArray) SubScalar_Inplace(b float64) {
	if a.dtype == Float32 {
		vek32.SubNumber_Inplace(a.Data.([]float32), float32(b))
	} else {
		vek.SubNumber_Inplace(a.Data.([]float64), b)
	}
}

// MulScalar_Inplace multiplies each element by a scalar: a *= b.
func (a *NdArray) MulScalar_Inplace(b float64) {
	if a.dtype == Float32 {
		vek32.MulNumber_Inplace(a.Data.([]float32), float32(b))
	} else {
		vek.MulNumber_Inplace(a.Data.([]float64), b)
	}
}

// DivScalar_Inplace divides each element by a scalar: a /= b.
func (a *NdArray) DivScalar_Inplace(b float64) {
	if a.dtype == Float32 {
		vek32.DivNumber_Inplace(a.Data.([]float32), float32(b))
	} else {
		vek.DivNumber_Inplace(a.Data.([]float64), b)
	}
}

// Abs_Inplace computes the absolute value in-place.
func (a *NdArray) Abs_Inplace() {
	if a.dtype == Float32 {
		vek32.Abs_Inplace(a.Data.([]float32))
	} else {
		vek.Abs_Inplace(a.Data.([]float64))
	}
}

// Neg_Inplace computes the negation in-place.
func (a *NdArray) Neg_Inplace() {
	if a.dtype == Float32 {
		vek32.Neg_Inplace(a.Data.([]float32))
	} else {
		vek.Neg_Inplace(a.Data.([]float64))
	}
}

// Sqrt_Inplace computes the square root in-place.
func (a *NdArray) Sqrt_Inplace() {
	if a.dtype == Float32 {
		vek32.Sqrt_Inplace(a.Data.([]float32))
	} else {
		vek.Sqrt_Inplace(a.Data.([]float64))
	}
}

// Round_Inplace rounds elements in-place.
func (a *NdArray) Round_Inplace() {
	if a.dtype == Float32 {
		vek32.Round_Inplace(a.Data.([]float32))
	} else {
		vek.Round_Inplace(a.Data.([]float64))
	}
}

// Floor_Inplace floors elements in-place.
func (a *NdArray) Floor_Inplace() {
	if a.dtype == Float32 {
		vek32.Floor_Inplace(a.Data.([]float32))
	} else {
		vek.Floor_Inplace(a.Data.([]float64))
	}
}

// Ceil_Inplace ceils elements in-place.
func (a *NdArray) Ceil_Inplace() {
	if a.dtype == Float32 {
		vek32.Ceil_Inplace(a.Data.([]float32))
	} else {
		vek.Ceil_Inplace(a.Data.([]float64))
	}
}

// CumSum_Inplace computes cumulative sum in-place.
func (a *NdArray) CumSum_Inplace() {
	if a.dtype == Float32 {
		vek32.CumSum_Inplace(a.Data.([]float32))
	} else {
		vek.CumSum_Inplace(a.Data.([]float64))
	}
}

// CumProd_Inplace computes cumulative product in-place.
func (a *NdArray) CumProd_Inplace() {
	if a.dtype == Float32 {
		vek32.CumProd_Inplace(a.Data.([]float32))
	} else {
		vek.CumProd_Inplace(a.Data.([]float64))
	}
}
