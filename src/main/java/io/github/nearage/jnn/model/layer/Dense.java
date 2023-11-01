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
package io.github.nearage.jnn.model.layer;

import io.github.nearage.jnn.input.Matrix;
import io.github.nearage.jnn.processing.Activation;
import io.github.nearage.jnn.processing.Layer;
import io.github.nearage.jnn.util.Matrices;

/**
 * Dense layer
 * 
 * @author Nearage <https://github.com/Nearage>
 */
public class Dense extends Layer {

    /**
     * Number of neurons
     */
    private final int neurs;
    /**
     * Activation functions
     */
    private final Activation[] activation;

    /**
     * Creates a dense layer with the specified neurs and activation function
     * 
     * @param neurs number of neurs
     * @param activation activation function
     */
    public Dense(int neurs, Activation[] activation) {
        this.neurs = neurs;
        this.activation = activation;
    }

    @Override
    public Matrix activate(Matrix input) throws Exception {
        if (this.weights == null) {
            this.weights = new Matrix(input.cols, neurs);
            this.weights.randomize(-1, 1);
            this.biases = new Matrix(1, neurs);
            this.biases.randomize(-1, 1);
        }

        Matrix z = Matrices.dot(input, this.weights);
        z.map((i, j) -> z.get(i, j) + this.biases.get(0, j));
        Matrix a = this.activation[0].apply(z);

        return a;
    }

    @Override
    public Matrix propagate(Matrix input) throws Exception {
        return this.activation[1].apply(input);
    }

}
