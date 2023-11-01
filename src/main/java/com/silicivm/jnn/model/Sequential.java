/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.silicivm.jnn.model;

import com.silicivm.jnn.input.Matrix;
import com.silicivm.jnn.processing.Layer;
import com.silicivm.jnn.processing.Loss;
import com.silicivm.jnn.processing.Model;
import com.silicivm.jnn.util.Matrices;

/**
 * Sequential model
 * 
 * @author vulgr
 */
public class Sequential extends Model {

    /**
     * Creates a Sequential model with the given layers
     * 
     * @param layers layers of the model
     */
    public Sequential(Layer... layers) {
        super(layers);
    }
    
    @Override
    public Matrix[] activate(Matrix input) throws Exception {
        // Array of activations
        Matrix[] result = new Matrix[this.layers.length + 1];

        // Set first element to input
        result[0] = input;

        // Iterate layers
        for (int l = 0; l < this.layers.length; l++) {
            result[l + 1] = this.layers[l].activate(result[l]);
        }

        // Return array of activations
        return result;
    }
    
    @Override
    public Matrix predict(Matrix input) throws Exception {
        // Activation of the model
        Matrix[] result = this.activate(input);

        // Return last activation
        return result[result.length - 1];
    }
    
    @Override
    public void train(
        int epochs,
        Matrix input,
        Matrix target,
        Loss[] loss,
        double learningRate
    ) throws Exception {
        System.out.println("Training..." + System.lineSeparator());

        // Iterate epochs
        for (int e = 0; e < epochs; e++) {
            // Activation of the model
            Matrix[] activations = this.activate(input);
            // Deltas for each layer
            Matrix[] deltas = new Matrix[this.layers.length + 1];
            // Last activation
            Matrix prediction = activations[activations.length - 1];
            // Last layer
            Layer lastLayer = this.layers[this.layers.length - 1];

            // Last layer deltas
            deltas[deltas.length - 1] = Matrices.mul(
                loss[1].apply(prediction, target),
                lastLayer.propagate(prediction)
            );

            // Iterate layers
            for (int l = this.layers.length - 1; l >= 0; l--) {
                // Layer deltas
                deltas[l] = Matrices.mul(
                    Matrices.dot(
                        deltas[l + 1],
                        this.layers[l].weights.transpose()
                    ),
                    this.layers[l].propagate(activations[l])
                );

                // Previous layer deltas
                Matrix delta = Matrices.dot(
                    activations[l].transpose(),
                    deltas[l + 1]
                );
                
                delta.apply(x -> x * learningRate);

                // Layer weights correction
                this.layers[l].weights = Matrices.sub(
                    this.layers[l].weights,
                    delta
                );

                // Previous deltas average
                double avg = deltas[l + 1].avg();

                // Layer biases correction
                this.layers[l].biases.apply(x -> x - avg * learningRate);
            }

            // Print status 10 times while training
            if ((e + 1) % (epochs / 10) == 0) {
                System.out.printf("Epoch %d error: %.8f%n",
                    e + 1,
                    loss[0].apply(prediction, target).peek()
                );
            }
        }

        System.out.println(
            System.lineSeparator()
            + "...done"
            + System.lineSeparator()
        );
    }
    
    @Override
    public void summary() throws Exception {
        StringBuilder layerDescription = new StringBuilder();

        int params = 0;

        for (Layer layer : this.layers) {
            if (layer.weights == null) {
                throw new Exception("Model not built");
            }

            layerDescription.append(String.format("  %s: %d params%n",
                layer.getClass().getSimpleName(),
                layer.weights.size + layer.biases.size
            ));

            params += layer.weights.size + layer.biases.size;
        }

        System.out.printf("%s%n"
            + " %d total trainable params%n"
            + " %d layers%n"
            + "%s%n",
            this.getClass().getSimpleName(),
            params,
            this.layers.length,
            layerDescription.toString()
        );
    }
}
