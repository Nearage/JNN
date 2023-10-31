/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.processing;

import com.silicivm.jnn.input.Matrix;
import com.silicivm.jnn.util.Matrices;

/**
 * Loss functions
 * 
 * @author vulgr
 */
public interface Loss {
    /**
     * Applies the loss function to the given input
     * 
     * @param input input data
     * @param target target data
     * 
     * @return the loss matrix
     * 
     * @throws Exception a base exception with an error message
     */
    public Matrix apply(Matrix input, Matrix target) throws Exception;

    /**
     * Mean Squared Error loss function
     */
    public static Loss[] MeanSquaredError = {
        (input, target) -> {
            Matrix result = new Matrix(input.rows, input.cols);

            result.map((i, j) -> Math.pow(input.get(i, j) - target.get(i, j), 2));

            result = result.reduce(0, 0, (y, x) -> y + x);
            result = result.reduce(1, 0, (y, x) -> y + x);
            
            result.apply(x -> x / input.size);

            return result;
        },
        (input, target) -> Matrices.sub(input, target)
    };
}
