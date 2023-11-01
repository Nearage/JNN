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
package io.github.nearage.jnn.input;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Matrix of doubles
 *
 * @author Nearage <https://github.com/Nearage>
 */
public class Matrix implements Serializable {

    /**
     * Number of rows
     */
    public final int rows;

    /**
     * Number of cols
     */
    public final int cols;

    /**
     * Size of the matrix
     */
    public final int size;

    /**
     * Data of the matrix
     */
    private final double[] data;

    /**
     * Creates a new matrix with the specified shape
     *
     * @param rows number of rows
     * @param cols number of cols
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.size = rows * cols;
        this.data = new double[size];
    }

    /**
     * Gets the value at the given index
     *
     * @param i row index
     * @param j col index
     *
     * @return the element at the given index
     */
    public double get(int i, int j) {
        return this.data[i * this.cols + j];
    }

    /**
     * Sets the element at the given index to de given value
     *
     * @param i row index
     * @param j col index
     * @param value new value
     */
    public void set(int i, int j, double value) {
        this.data[i * this.cols + j] = value;
    }

    /**
     * Gets the value of the first element
     *
     * @return the value of the first element
     */
    public double peek() {
        return this.get(0, 0);
    }

    /**
     * Performs an action on each index in the matrix
     *
     * @param action the action to perform
     */
    public void iterate(BiConsumer<Integer, Integer> action) {
        for (int s = 0; s < this.size; s++) {
            int i = s / this.cols;
            int j = s % this.cols;

            action.accept(i, j);
        }
    }

    /**
     * Performs an action on each element in the matrix
     *
     * @param action the action to perform
     */
    public void foreach(Consumer<Double> action) {
        this.iterate((i, j) -> action.accept(this.get(i, j)));
    }

    /**
     * Sets the value of each index in the matrix to the result of applying the
     * given function to that index
     *
     * @param function the function to apply
     */
    public void map(BiFunction<Integer, Integer, Double> function) {
        this.iterate((i, j) -> this.set(i, j, function.apply(i, j)));
    }

    /**
     * Sets the value of each element in the matrix to the result of applying
     * the given function to that element
     *
     * @param function the function to apply
     */
    public void apply(Function<Double, Double> function) {
        this.map((i, j) -> function.apply(this.get(i, j)));
    }

    /**
     * Randomizes the elements in the matrix within the given bounds, both
     * included
     *
     * @param origin lower bound
     * @param bound upper bound
     */
    public void randomize(int origin, int bound) {
        Random rng = new Random();

        this.apply(x -> rng.nextDouble(origin, bound));
    }

    /**
     * Transposes the matrix
     *
     * @return the transposed matrix
     */
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);

        result.map((i, j) -> this.get(j, i));

        return result;
    }

    /**
     * Reduces the matrix along the specified axis applying the given identity
     * and function
     *
     * @param axis axis to reduce
     * @param identity identity value
     * @param function function to apply
     *
     * @return the reduced matrix
     */
    public Matrix reduce(
            int axis,
            double identity,
            BiFunction<Double, Double, Double> function
    ) {
        Matrix result = new Matrix(
                axis == 0 ? this.rows : 1,
                axis == 1 ? this.cols : 1
        );

        int range = axis == 0 ? this.cols : this.rows;

        result.map((i, j) -> {
            double value = identity;

            for (int a = 0; a < range; a++) {
                if (axis == 0) {
                    value = function.apply(value, this.get(i, a));
                }
                if (axis == 1) {
                    value = function.apply(value, this.get(a, j));
                }
            }

            return value;
        });

        return result;

    }

    /**
     * Reduces the matrix to a single value applying the given identity and
     * function
     *
     * @param identity identity value
     * @param function function to apply
     *
     * @return the reduced value
     */
    public double reduce(
            double identity,
            BiFunction<Double, Double, Double> function
    ) {
        Matrix result = this.reduce(0, identity, function)
                            .reduce(1, identity, function);

        return result.peek();
    }

    /**
     * Gets the min value in the matrix
     *
     * @return the min value
     */
    public double min() {
        return this.reduce(Double.MAX_VALUE, (y, x) -> Math.min(y, x));
    }

    /**
     * Gets the max value in the matrix
     *
     * @return the max value
     */
    public double max() {
        return this.reduce(Double.MIN_VALUE, (y, x) -> Math.max(y, x));
    }

    /**
     * Gets the sum of all values in the matrix
     *
     * @return the sum value
     */
    public double sum() {
        return this.reduce(0, (y, x) -> y + x);
    }

    /**
     * Gets the average value of the matrix
     *
     * @return the min value
     */
    public double avg() {
        return this.sum() / this.size;
    }

    /**
     * Prints a description of the matrix
     */
    public void describe() {
        System.out.printf(
                "%s%n"
                + " rows: %d%n"
                + " cols: %d%n"
                + " size: %d%n"
                + " min:  %.8f%n"
                + " max:  %.8f%n"
                + " sum:  %.8f%n"
                + " avg:  %.8f%n%n",
                getClass().getSimpleName(),
                this.rows,
                this.cols,
                this.size,
                this.min(),
                this.max(),
                this.sum(),
                this.avg()
        );
    }

    /**
     * Prints a summary of the matrix
     */
    public void summary() {
        System.out.printf(
                "%s%n"
                + " rows: %d%n"
                + " cols: %d%n"
                + " size: %d%n%n",
                getClass().getSimpleName(),
                this.rows,
                this.cols,
                this.size
        );
    }

    /**
     * Prints every value of the matrix
     */
    public void print() {
        this.iterate((i, j) -> {
            System.out.printf(
                    "%11.8f %s",
                    this.get(i, j),
                    this.cols - j == 1 ? System.lineSeparator() : ""
            );
        });

        System.out.println("");
    }

    /**
     * Saves the matrix to a file in the specified path
     *
     * @param path path
     *
     * @throws Exception a base exception with an error message
     */
    public void save(String path) throws Exception {
        FileOutputStream fos = new FileOutputStream(path);

        try (ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this);
            oos.flush();
        }
    }

    /**
     * Loads a matrix from a file in the specified path
     *
     * @param path path
     *
     * @return the loaded matrix
     *
     * @throws Exception a base exception with an error message
     */
    public static Matrix load(String path) throws Exception {
        FileInputStream fis = new FileInputStream(path);

        try (ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (Matrix) ois.readObject();
        }
    }
}
