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
package io.github.nearage.jnn.processing;

import io.github.nearage.jnn.input.Matrix;

/**
 * Activation functions
 * 
 * @author Nearage <https://github.com/Nearage>
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
