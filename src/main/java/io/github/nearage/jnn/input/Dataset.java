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

/**
 * TODO: Dataset
 * 
 * @author Nearage <https://github.com/Nearage>
 */
public class Dataset {
    public final Matrix[] inputs;
    public final Matrix[] targets;

    public Dataset(Matrix[] inputs, Matrix[] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public static Dataset generate_sum(int inputs, int params, int batchSize) throws Exception {
        int batches = (int) inputs / batchSize;
        int reminder = inputs - batchSize * batches;

        Matrix[] in = new Matrix[batches + (reminder > 0 ? 1 : 0)];
        Matrix[] tg = new Matrix[batches + (reminder > 0 ? 1 : 0)];

        for (int n = 0; n < in.length; n++) {
            if (reminder > 0 && n == in.length - 1) {
                batchSize = reminder;
            }

            Matrix input = new Matrix(batchSize, params);
            input.randomize(-1, 1);

            Matrix target = new Matrix(batchSize, 2);
            Matrix sum = input.reduce(0, 0d, (y, x) -> y + x);

            target.map((i, j)
                -> (sum.get(i, 0) > 0d)
                ? (j == 0 ? 1d : 0d)
                : (j == 1 ? 1d : 0d)
            );

            in[n] = input;
            tg[n] = target;
        }

        return new Dataset(in, tg);
    }

    public static Dataset batch() {
        return null;
    }
}
