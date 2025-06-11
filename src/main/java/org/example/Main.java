package org.example;

public class Main {
    public static void main(String[] args) {
        // Simulated normalized training data for 5 color classes
        double[][] trainData = {
                {0.1, 0.2, 0.3, 0.4},     // blue
                {0.12, 0.18, 0.33, 0.42}, // blue
                {0.9, 0.8, 0.7, 0.6},     // red
                {0.88, 0.82, 0.73, 0.61}, // red
                {0.5, 0.5, 0.5, 0.5},     // green
                {0.52, 0.48, 0.49, 0.51}, // green
                {0.3, 0.7, 0.6, 0.2},     // yellow
                {0.31, 0.69, 0.61, 0.19}, // yellow
                {11.2, 10.3, 13.4, 15.2}  // black
        };

        String[] trainLabels = {
                "blue", "blue",
                "red", "red",
                "green", "green",
                "yellow", "yellow",
                "black"
        };

        // Simulated test data for each class
        double[][] testData = {
                {0.11, 0.21, 0.29, 0.41}, // blue
                {0.89, 0.81, 0.72, 0.62}, // red
                {0.51, 0.49, 0.48, 0.52}, // green
                {0.32, 0.68, 0.62, 0.21}, // yellow
                {11.00000, 10.01, 13.1, 15.3} // black
        };

        String[] testLabels = {
                "blue",
                "red",
                "green",
                "yellow",
                "black"
        };

        // Validate feature size
        if (trainData[0].length != testData[0].length) {
            System.out.println("Warning: Feature length mismatch between training and test data.");
            return;
        }

        // Initialize and train LVQ
        LVQClassifier lvq = new LVQClassifier(trainData, trainLabels);

        // Predict and calculate accuracy
        int correct = 0;
        for (int i = 0; i < testData.length; i++) {
            String predicted = lvq.predict(testData[i]);
            System.out.println("Test " + (i + 1) + " | Predicted: " + predicted + " | Actual: " + testLabels[i]);
            if (predicted.equals(testLabels[i])) {
                correct++;
            }
        }

        double accuracy = (double) correct / testData.length;
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
}
