use std::ops::{Add, Sub, Mul};
use ndarray::*;




/**
 * Return the cartesian product of two 1D arrays. If the argument arrays have
 * elements types T and U, the resulting array has element type (T, U). If the
 * argument arrays have lengths n1, n2 the resulting array has shape (n1, n2).
 */
pub fn cartesian_product2<T: Copy, U: Copy>(x: Array1<T>, y: Array1<U>) -> Array2<(T, U)>
{
    Array::from_shape_fn((x.len(), y.len()), |(i, j)| (x[i], y[j]))
}




/**
 * Return the cartesian product of three 1D arrays. If the argument arrays have
 * elements types T, U, and V, the resulting array has element type (T, U, V).
 * If the argument arrays have lengths n1, n2, and n3, the resulting array has
 * shape (n1, n2, n3).
 */
pub fn cartesian_product3<T: Copy, U: Copy, V: Copy>(x: Array1<T>, y: Array1<U>, z: Array1<V>) -> Array3<(T, U, V)>
{
    Array::from_shape_fn((x.len(), y.len(), z.len()), |(i, j, k)| (x[i], y[j], z[k]))
}




/**
 * Map a function over the adjacent elements of an array. If the array has
 * element type T, the function must map two T's to a U. The resulting array has
 * element type U and is one element shorter along the given axis.
 */
pub fn map_stencil2<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-1));
    let b = x.slice_axis(axis, Slice::from(1..n-0));
    azip![a, b].apply_collect(f)
}




/**
 * Map a function over the adjacent elements of an array. If the array has
 * element type T, the function must map three T's to a U. The resulting array
 * has element type U and is two elements shorter along the given axis.
 */
pub fn map_stencil3<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-2));
    let b = x.slice_axis(axis, Slice::from(1..n-1));
    let c = x.slice_axis(axis, Slice::from(2..n-0));
    azip![a, b, c].apply_collect(f)
}




/**
 * Map a function over the adjacent elements of an array. If the array has
 * element type T, the function must map four T's to a U. The resulting array
 * has element type U and is three elements shorter along the given axis.
 */
pub fn map_stencil4<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T, &T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-3));
    let b = x.slice_axis(axis, Slice::from(1..n-2));
    let c = x.slice_axis(axis, Slice::from(2..n-1));
    let d = x.slice_axis(axis, Slice::from(3..n-0));
    azip![a, b, c, d].apply_collect(f)
}




/**
 * Obtain the mean of adjacent elements along a given axis. The resulting array
 * is one element shorter on that axis.
 */
pub fn adjacent_mean<T, U, P, D>(x: &ArrayBase<P, D>, axis: Axis) -> Array<U, D>
    where
        T: Copy + Add<Output=U>,
        U: Copy + Mul<f64, Output=U>,
        P: RawData<Elem=T> + Data,
        D: Dimension,
{
    map_stencil2(x, axis, |&a, &b| (a + b) * 0.5)
}




/**
 * Obtain the difference of adjacent elements along a given axis. The resulting
 * array is one element shorter on that axis.
 */
pub fn adjacent_diff<T, U, P, D>(x: &ArrayBase<P, D>, axis: Axis) -> Array<U, D>
    where
        T: Copy + Sub<Output=U>,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
{
    map_stencil2(x, axis, |&a, &b| b - a)
}




/**
 * Return a 2d array which is extended periodically on both axes by a given
 * number of 'guard' zones ng. If ng = 3 and the array given has size (100, 100)
 * then the resulting array has size (106, 106).
 */
 pub fn extend_periodic_2d<T, P>(a: ArrayBase<P, Ix2>, ng: usize) -> Array2<T>
    where
        T: Copy,
        P: RawData<Elem=T> + Data
{
    let (ni, nj) = a.dim();
    let extended_shape = (ni + 2 * ng, nj + 2 * ng);

    Array::from_shape_fn(extended_shape, |(mut i, mut j)| -> T
    {
        if i < ng {
            i += ni;
        } else if i >= ni + ng {
            i -= ni;
        }
        if j < ng {
            j += nj;
        } else if j >= nj + ng {
            j -= nj;
        }
        unsafe {
            *a.uget([i - ng, j - ng])
        }
    })
}




/**
 * Return a 2d array which is extended by a given number of 'guard' zones on
 * each edge, using the default value for the array type T, which must be Default.
 *
 * * `a`   - an ndarray
 * * `gli` - number of guard zones to add to the west edge of the array
 * * `gri` - number of guard zones to add to the east edge of the array
 * * `glj` - number of guard zones to add to the south edge of the array
 * * `grj` - number of guard zones to add to the north edge of the array
 */
pub fn extend_default_2d<T:, P>(a: ArrayBase<P, Ix2>, gli: usize, gri: usize, glj: usize, grj: usize) -> Array2<T>
    where
        T: Copy + Default,
        P: RawData<Elem=T> + Data
{
    if gli == 0 && gri == 0 && glj == 0 && grj == 0 {
        return a.to_owned();
    }

    let (ni, nj) = a.dim();
    let extended_shape = (ni + gli + gri, nj + glj + grj);

    Array::from_shape_fn(extended_shape, |(i, j)| -> T
    {
        if i < gli {
            return T::default();
        } else if i >= ni + gli {
            return T::default();
        }
        if j < glj {
            return T::default();
        } else if j >= nj + glj {
            return T::default();
        }
        unsafe {
            *a.uget([i - gli, j - glj])
        }
    })
}




/**
 * Return a 2d array which is extended by a given number of 'guard' zones on
 * each edge. The extension is done to the array at the middle of a 3x3
 * fixed-length array of nd-arrays. The surrounding 8 nd-arrays can have
 * distinct shapes, but must have the same length along any shared edges. In
 * other words, the array at (1, 2), lying due north of the array to be
 * extended, must have the same length on the i-axis as the array to be
 * extended. Length on the j-axis must be greater than or equal to the number of
 * guard zones grj on the north edge (assuming you have j increasing from bottom
 * to top in your head).
 *
 * * `a`   - 3x3 fixed-length array of nd-arrays, the middle (1, 1) of which is
 *   to be extended
 * * `gli` - number of guard zones to add to the west edge of the array
 * * `gri` - number of guard zones to add to the east edge of the array
 * * `glj` - number of guard zones to add to the south edge of the array
 * * `grj` - number of guard zones to add to the north edge of the array
 */
pub fn extend_from_neighbor_arrays_2d<T, P>(a: &[[ArrayBase<P, Ix2>; 3]; 3], gli: usize, gri: usize, glj: usize, grj: usize) -> Array2<T>
    where
        T: Copy,
        P: RawData<Elem=T> + Data,
{
    fn block_index(i: usize, gl: usize, n0: usize, gr: usize) -> usize
    {
        if i < gl {
            0
        } else if i < gl + n0 {
            1
        } else if i < gl + n0 + gr {
            2
        } else {
            panic!();
        }
    }
    let (ni, nj) = a[1][1].dim();
    let extended_shape = (ni + gli + gri, nj + glj + grj);

    Array::from_shape_fn(extended_shape, |(i, j)| -> T
    {
        let bi = block_index(i, gli, ni, gri);
        let bj = block_index(j, glj, nj, grj);
        let i0 = match bi {
            0 => i + a[0][bj].dim().0 - gli,
            1 => i - gli,
            2 => i - a[1][bj].dim().0 - gli,
            _ => unreachable!(),
        };
        let j0 = match bj {
            0 => j + a[bi][0].dim().1 - glj,
            1 => j - glj,
            2 => j - a[bi][1].dim().1 - glj,
            _ => unreachable!(),
        };
        unsafe {
            *a[bi][bj].uget([i0, j0])
        }
    })
}




/**
 * Extends 3 by 3 fixed-length array types, [[T; 3]; 3], with a functor map.
 * This trait is defined here because it composes nicely with the
 * `extend_from_neighbor_arrays_2d` function.
 */
pub trait MapArray3by3
{
    type Elem;
    fn map<F: Fn(&Self::Elem) -> U, U>(&self, f: F) -> [[U; 3]; 3];
}

impl<T> MapArray3by3 for [[T; 3]; 3]
{
    type Elem = T;
    fn map<F, U>(&self, f: F) -> [[U; 3]; 3] where F: Fn(&Self::Elem) -> U {
        [
            [f(&self[0][0]), f(&self[0][1]), f(&self[0][2])],
            [f(&self[1][0]), f(&self[1][1]), f(&self[1][2])],
            [f(&self[2][0]), f(&self[2][1]), f(&self[2][2])],
        ]
    }
}




// ============================================================================
#[cfg(test)]
mod tests
{
    use ndarray::Array2;

    #[test]
    fn extend_default_works()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        assert_eq!(crate::extend_default_2d(x.clone(), 2, 2, 2, 2).dim(), (14, 14));
        assert_eq!(crate::extend_default_2d(x.clone(), 0, 1, 2, 3).dim(), (11, 15));
    }

    #[test]
    fn extend_from_neighbor_arrays_works_with_uniformly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let a = [
            [x.clone(), x.clone(), x.clone()],
            [x.clone(), x.clone(), x.clone()],
            [x.clone(), x.clone(), x.clone()]
        ];
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 2, 2, 2, 2).dim(), (14, 14));
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 3, 3, 2, 2).dim(), (16, 14));
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 2, 2, 3, 3).dim(), (14, 16));
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 3, 1, 2, 2).dim(), (14, 14));
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 2, 2, 3, 1).dim(), (14, 14));
    }

    #[test]
    fn extend_from_neighbor_arrays_works_with_non_uniformly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let y = Array2::<f64>::zeros(( 3,  3)).to_shared(); // these are in the corner

        let a = [
            [y.clone(), x.clone(), y.clone()],
            [x.clone(), x.clone(), x.clone()],
            [y.clone(), x.clone(), y.clone()],
        ];
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 1, 1, 1, 1).dim(), (12, 12));
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 1, 2, 1, 2).dim(), (13, 13));
    }

    #[test]
    fn extend_from_neighbor_arrays_works_with_non_zero_length_arrays()
    {
        let w = Array2::<f64>::zeros(( 2, 10)).to_shared();
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let y = Array2::<f64>::zeros(( 0,  0)).to_shared();

        let a = [
            [y.clone(), w.clone(), y.clone()],
            [y.clone(), x.clone(), y.clone()],
            [y.clone(), w.clone(), y.clone()],
        ];
        assert_eq!(crate::extend_from_neighbor_arrays_2d(&a, 2, 2, 0, 0).dim(), (14, 10));
    }

    #[test]
    #[should_panic]
    fn extend_from_neighbor_arrays_panics_with_wrongly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let y = Array2::<f64>::zeros(( 3,  3)).to_shared();

        let a = [
            [y.clone(), y.clone(), y.clone()],
            [x.clone(), x.clone(), x.clone()],
            [y.clone(), y.clone(), y.clone()],
        ];
        crate::extend_from_neighbor_arrays_2d(&a, 1, 1, 1, 1);
    }
}
