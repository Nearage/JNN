/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Interface.java to edit this template
 */
package com.silicivm.jnn.util;

import com.silicivm.jnn.input.Matrix;

/**
 * Matrices operations
 * 
 * @author vulgr
 */
public interface Matrices {

    /**
     * Performs a dot product of two matrices
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
