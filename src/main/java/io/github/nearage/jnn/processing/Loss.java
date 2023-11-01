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
import io.github.nearage.jnn.util.Matrices;

/**
 * Loss functions
 * 
 * @author Nearage <https://github.com/Nearage>
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
