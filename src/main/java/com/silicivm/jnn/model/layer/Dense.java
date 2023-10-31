/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.model.layer;

import com.silicivm.jnn.input.Matrix;
import com.silicivm.jnn.processing.Activation;
import com.silicivm.jnn.processing.Layer;
import com.silicivm.jnn.util.Matrices;

/**
 * Dense layer
 * 
 * @author vulgr
 */
public class Dense extends Layer {

    private final int neurs;
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
