/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.processing;

import com.silicivm.jnn.input.Matrix;

/**
 * Activation functions
 * 
 * @author vulgr
 */
public interface Activation {

    /**
     * Applies the activation function to the given input
     * 
     * @param input input data
     * 
     * @return the activation matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public Matrix apply(Matrix input) throws Exception;
    
    /**
     * Rectified Linear Unit activation function
     */
    public static Activation[] RectifiedLinearUnit = {
        input -> {
            Matrix result = new Matrix(input.rows, input.cols);

            result.map((i, j) -> Math.max(0d, input.get(i, j)));

            return result;
        },
        input -> {
            Matrix result = new Matrix(input.rows, input.cols);

            result.map((i, j) -> input.get(i, j) > 0d ? 1d : 0d);

            return result;
        }
    };

    /**
     * Sigmoid activation function
     */
    public static Activation[] Sigmoid = {
        input -> {
            Matrix result = new Matrix(input.rows, input.cols);

            result.map((i, j) -> 1d / (1d + Math.exp(-input.get(i, j))));

            return result;
        },
        input -> {
            Matrix result = new Matrix(input.rows, input.cols);

            result.map((i, j) -> input.get(i, j) * (1 - input.get(i, j)));

            return result;
        }
    };

    /**
     * Softmax activation function
     */
    public static Activation[] Softmax = {
        input -> {
            Matrix max = input.reduce(0, Double.MIN_VALUE, (y, x) -> Math.max(y, x));

            Matrix exp = new Matrix(input.rows, input.cols);

            exp.map((i, j) -> Math.exp(input.get(i, j) - max.get(i, 0)));

            Matrix sum = exp.reduce(0, 0d, (y, x) -> y + x);

            exp.map((i, j) -> exp.get(i, j) / sum.get(i, 0));

            return exp;
        },
        Activation.RectifiedLinearUnit[1]
    };
}
