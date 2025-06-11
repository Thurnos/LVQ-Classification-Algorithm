package org.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LVQClassifier {

    private double[][] prototypes;
    private String[] protoLabels;
    private double alpha = 0.1;
    private int epochs = 20;

    public LVQClassifier(double[][] trainData, String[] trainLabels) {
        initializePrototypes(trainData, trainLabels);
        train(trainData, trainLabels);
    }

    // 1. Initialize one prototype per class (mean of each class)
    private void initializePrototypes(double[][] data, String[] labels) {
        Map<String, List<double[]>> classData = new HashMap<>();
        for (int i = 0; i < data.length; i++) {
            classData.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(data[i]);
        }

        prototypes = new double[classData.size()][data[0].length];
        protoLabels = new String[classData.size()];
        int index = 0;

        for (Map.Entry<String, List<double[]>> entry : classData.entrySet()) {
            protoLabels[index] = entry.getKey();
            double[] mean = new double[data[0].length];
            for (double[] sample : entry.getValue()) {
                for (int j = 0; j < sample.length; j++) {
                    mean[j] += sample[j];
                }
            }
            for (int j = 0; j < mean.length; j++) {
                mean[j] /= entry.getValue().size();
            }
            prototypes[index] = mean;
            index++;
        }
    }

    // 2. Train the model
    private void train(double[][] data, String[] labels) {
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < data.length; i++) {
                double[] x = data[i];
                String label = labels[i];

                int winner = findNearestPrototype(x);
                if (protoLabels[winner].equals(label)) {
                    for (int j = 0; j < x.length; j++) {
                        prototypes[winner][j] += alpha * (x[j] - prototypes[winner][j]);
                    }
                } else {
                    for (int j = 0; j < x.length; j++) {
                        prototypes[winner][j] -= alpha * (x[j] - prototypes[winner][j]);
                    }
                }
            }
            alpha *= 0.95;
        }
    }

    // 3. Predict a new input
    public String predict(double[] input) {
        int winner = findNearestPrototype(input);
        return protoLabels[winner];
    }

    // 4. Nearest prototype based on Euclidean distance
    private int findNearestPrototype(double[] x) {
        double minDist = Double.MAX_VALUE;
        int best = -1;
        for (int i = 0; i < prototypes.length; i++) {
            double dist = 0.0;
            for (int j = 0; j < x.length; j++) {
                dist += Math.pow(x[j] - prototypes[i][j], 2);
            }
            if (dist < minDist) {
                minDist = dist;
                best = i;
            }
        }
        return best;
    }
}
