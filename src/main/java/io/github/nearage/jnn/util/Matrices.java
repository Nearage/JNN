/*
 * Copyright (C) 2023 Nearage <https://github.com/Nearage>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package io.github.nearage.jnn.util;

import io.github.nearage.jnn.input.Matrix;

/**
 * Matrix utils
 * 
 * @author Nearage <https://github.com/Nearage>
 */
public interface Matrices {

    /**
     * Performs the dot product of two matrices
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return the resulting matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public static Matrix dot(Matrix a, Matrix b) throws Exception {
        if (a.cols != b.rows) {
            throw new Exception(String.format(
                "Shape mismatch in dot, a = (%d, %d) and b = (%d, %d)",
                a.rows,
                a.cols,
                b.rows,
                b.cols
            ));
        }

        Matrix result = new Matrix(a.rows, b.cols);

        result.map((i, j) -> {
            double sum = 0;

            for (int k = 0; k < a.cols; k++) {
                sum += a.get(i, k) * b.get(k, j);
            }

            return sum;
        });

        return result;
    }

    /**
     * Adds matrix b to matrix a
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return the resulting matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public static Matrix add(Matrix a, Matrix b) throws Exception {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new Exception(String.format(
                "Shape mismatch in add, a = (%d, %d) and b = (%d, %d)",
                a.rows,
                a.cols,
                b.rows,
                b.cols
            ));
        }

        Matrix result = new Matrix(a.rows, b.cols);

        result.map((i, j) -> a.get(i, j) + b.get(i, j));

        return result;
    }

    /**
     * Substracts matrix b to matrix a
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return the resulting matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public static Matrix sub(Matrix a, Matrix b) throws Exception {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new Exception(String.format(
                "Shape mismatch in sub, a = (%d, %d) and b = (%d, %d)",
                a.rows,
                a.cols,
                b.rows,
                b.cols
            ));
        }

        Matrix result = new Matrix(a.rows, b.cols);

        result.map((i, j) -> a.get(i, j) - b.get(i, j));

        return result;
    }

    /**
     * Multyplies matrix a by matrix b
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return the resulting matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public static Matrix mul(Matrix a, Matrix b) throws Exception {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new Exception(String.format(
                "Shape mismatch in mul, a = (%d, %d) and b = (%d, %d)",
                a.rows,
                a.cols,
                b.rows,
                b.cols
            ));
        }

        Matrix result = new Matrix(a.rows, b.cols);

        result.map((i, j) -> a.get(i, j) * b.get(i, j));

        return result;
    }
}
