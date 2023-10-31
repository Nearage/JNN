/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.processing;

import com.silicivm.jnn.input.Matrix;

/**
 * Model
 * 
 * @author vulgr
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
