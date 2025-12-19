# NDVek

NDVek is a high-performance Go library for multi-dimensional array operations, powered by the `vek` SIMD backend. It supports both `float64` and `float32` data types, automatic broadcasting, and a some common operations.

## Methods

| Category | Method | Description |
|---|---|---|
| **Creation** | `NewNdArray` | Creates a new array from a shape and data (`[]float64` or `[]float32`). |
| | `Zeros` | Creates an array of zeros with the specified shape. |
| | `Linspace` | Generates linearly spaced values. |
| **Arithmetic** | `Add`, `Subtract` | Element-wise addition and subtraction (supports broadcasting). |
| | `Multiply`, `Divide` | Element-wise multiplication and division (supports broadcasting). |
| | `AddScalar`, `SubScalar` | Scalar addition and subtraction. |
| | `MulScalar`, `DivScalar` | Scalar multiplication and division. |
| **Inplace Arithmetic** | `Add_Inplace`, `Subtract_Inplace` | Element-wise in-place addition and subtraction (requires equal shapes). |
| | `Multiply_Inplace`, `Divide_Inplace` | Element-wise in-place multiplication and division (requires equal shapes). |
| | `AddScalar_Inplace`, `SubScalar_Inplace` | Scalar in-place addition and subtraction. |
| | `MulScalar_Inplace`, `DivScalar_Inplace` | Scalar in-place multiplication and division. |
| | `Abs_Inplace`, `Neg_Inplace` | Element-wise in-place absolute value and negation. |
| | `Sqrt_Inplace`, `Round_Inplace` | Element-wise in-place square root and rounding. |
| | `Floor_Inplace`, `Ceil_Inplace` | Element-wise in-place floor and ceil. |
| | `CumSum_Inplace`, `CumProd_Inplace` | In-place cumulative sum and product. |
| **Broadcasting** | `ApplyOp` | Applies a custom binary function with broadcasting (promotes to `float64`). |
| | `ApplyHadamardOp` | Applies a custom unary function element-wise (promotes to `float64`). |
| **Aggregation** | `Sum` | Sum of all elements. |
| | `Mean` | Arithmetic mean of all elements. |
| | `Min`, `Max` | Minimum and maximum values. |
| | `Prod` | Product of all elements. |
| **Unary Ops** | `Abs` | Element-wise absolute value. |
| | `Neg` | Element-wise negation. |
| | `Sqrt` | Element-wise square root. |
| | `Round`, `Floor`, `Ceil` | Element-wise rounding operations. |
| | `CumSum` | Cumulative sum. |
| | `CumProd` | Cumulative product. |
| **Manipulation** | `Reshape` | Changes the shape of the array. |
| | `InsertAxis` | Inserts a new axis at the specified position. |
| | `Get` | Retrieves an element at a specific index. |
| | `Shape` | Returns the shape of the array. |
| | `DType` | Returns the data type (`Float32`, `Float64`, or `Bool`). |
| **Boolean Logic** | `Eq`, `Neq` | Element-wise equality/inequality comparison (returns `Bool` array). |
| | `Lt`, `Lte`, `Gt`, `Gte` | Element-wise comparison (returns `Bool` array). |
| | `And`, `Or`, `Xor` | Element-wise logical operations (requires `Bool` arrays). |
| | `Not` | Element-wise logical NOT (requires `Bool` array). |
| | `Any`, `All` | Returns true if any/all elements are true (requires `Bool` array). |

## Usage Examples

### Basic Arithmetic and Broadcasting

```go
package main

import (
    "fmt"
    "github.com/mederrata/ndvek"
)

func main() {
    // Create a 2x3 matrix
    shape := []int{2, 3}
    data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    arr, _ := ndvek.NewNdArray(shape, data)

    // Create a scalar (broadcastable)
    scalarArr, _ := ndvek.NewNdArray([]int{1}, []float64{10.0})

    // Add scalar to every element
    result, _ := ndvek.Add(arr, scalarArr)
    
    // Result is still 2x3
    // [11.0, 12.0, 13.0]
    // [14.0, 15.0, 16.0]
    fmt.Println(result.Mean()) // 13.5
}
```

### Boolean Logic

```go
// Create two arrays
a, _ := ndvek.NewNdArray([]int{4}, []float64{1, 2, 3, 4})
b, _ := ndvek.NewNdArray([]int{4}, []float64{2, 2, 2, 2})

// Compare them
// Lt: [1<2, 2<2, 3<2, 4<2] -> [T, F, F, F]
lt, _ := ndvek.Lt(a, b) 

// Check if any element is true
fmt.Println(lt.Any()) // true
```

> **Note**: For more comprehensive examples, please refer to the `ndarray_test.go` file in the source code.



