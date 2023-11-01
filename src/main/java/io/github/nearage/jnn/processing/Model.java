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
 * Model
 * 
 * @author Nearage <https://github.com/Nearage>
 */
public abstract class Model {

    /**
     * Layers of the model
     */
    public final Layer[] layers;

    /**
     * Creates a model with the given layers
     * 
     * @param layers layers of the model
     */
    public Model(Layer... layers) {
        this.layers = layers;
    }

    /**
     * Generates an activation for each layer of the model
     * 
     * @param input input data
     * 
     * @return array of activations
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract Matrix[] activate(Matrix input) throws Exception;

    /**
     * Generates output predictions for the input samples
     * 
     * @param input input samples
     * 
     * @return predictions
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract Matrix predict(Matrix input) throws Exception;

    /**
     * Trains the model for a fixed number of epochs
     * 
     * @param epochs number of epochs to train
     * @param input input data
     * @param target target data
     * @param loss loss function
     * @param learningRate learing rate
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract void train(
        int epochs,
        Matrix input,
        Matrix target,
        Loss[] loss,
        double learningRate
    ) throws Exception;
    
//    public abstract void train(
//        int epochs,
//        Matrix[] inputs,
//        Matrix[] targets,
//        Loss[] loss,
//        double learningRate
//    ) throws Exception;
    
    /**
     * Prints a string summary of the model
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract void summary() throws Exception;
}
