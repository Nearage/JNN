/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.processing;

import com.silicivm.jnn.input.Matrix;


/**
 * Layer
 * 
 * @author vulgr
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
