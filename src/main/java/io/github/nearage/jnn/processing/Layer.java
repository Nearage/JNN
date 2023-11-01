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
 * Layer
 * 
 * @author Nearage <https://github.com/Nearage>
 */
public abstract class Layer {
    /**
     * Weights of the layer
     */
    public Matrix weights;
    
    /**
     * Biases of the layer
     */
    public Matrix biases;
    
    /**
     * Activates the layer for the given input
     * 
     * @param input input data
     * 
     * @return the activation of the layer
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract Matrix activate(Matrix input) throws Exception;
    
    /**
     * Propagates the activation of the layer for the given input
     * 
     * @param input input data
     * 
     * @return the propagation of the layer
     * 
     * @throws Exception a base exception with an error message
     */
    public abstract Matrix propagate(Matrix input) throws Exception;
}
