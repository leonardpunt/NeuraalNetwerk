
import images.ImageReader;

import java.util.Calendar;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import neuraalnetwork.Connection;
import neuraalnetwork.Layer;
import neuraalnetwork.NeuralNetwork;
import neuraalnetwork.Neuron;

public class Run {

    static void threadMessage(String message) {
        String threadName =
                Thread.currentThread().getName();
        System.out.format("%s: %s%n",
                threadName,
                message);
    }

    private static class BackPropagateThread implements Runnable {

        int from, to;
        double learningRate;
        NeuralNetwork nn;

        // Constructor for BackPropagateThread since you can't give
        // run() any parameters.
        public BackPropagateThread(int from, int to, double learningRate, NeuralNetwork nn) {
            this.from = from;
            this.to = to;
            this.nn = nn;
            this.learningRate = learningRate;
        }

        public void run() {
            threadMessage("Thread is starting: from "
                    + from
                    + " to "
                    + to);
            // ImageReader is 
            ImageReader ir = new ImageReader();
            for (int i = this.from + 1; i < this.to; i++) {
                double[] image = ir.readImage(i, ir.getTrainingSet());
                int label = ir.readLabel(i, ir.getTrainingSet());
                double[] actualOutput = nn.forwardPropagate(image);
                nn.backPropagate(actualOutput, getTargetOutput(label),
                        learningRate);
                //threadMessage("reports last: " + i);
            }
            threadMessage("Thread ended");
        }
    }

    public static void main(String[] args) {
        Date timeStarted = Calendar.getInstance().getTime();
        System.out.println("--- Started at " + timeStarted + " ---");

        ImageReader ir = new ImageReader();

        /*
         * RESULTS
         * 
         * numberNeuronsHiddenLayer : 10 0.008 = 27%; 23%
         * 
         * numberNeuronsHiddenLayer : 50 
         * 0.05 = 51%; 53% 
         * 0.03 = 60%; 57% 
         * 0.01 = 53%; 63%; 59% 
         * 0.008 = 57%; 65%; 66%; 63% 
         * 0.005 = 68%; 67%; 57%
         * 
         * numberNeuronsHiddenLayer : 70 
         * 0.008 = 57%; 58
         * 
         * numberNeuronsHiddenLayer : 100 
         * 0.008 = 40%; 57%; 56%
         * 
         * numberNeuronsHiddenLayer : 150 
         * 0.008 = 53%
         * 
         * numberNeuronsHiddenLayer = 180;
         * 0.00005 = 18.66%
         * 0.0005 = 16.55%
         */
        int numberNeuronsHiddenLayer = 50;
        double learningRate = 0.005;

        // Initialize network
        NeuralNetwork nn = new NeuralNetwork(numberNeuronsHiddenLayer);

        // Get validation set
        int sizeValidationSet = 1000;
        List<Integer> indicesValidationSet = new LinkedList<Integer>();
        for (int i = 1; i <= sizeValidationSet; i++) {
            indicesValidationSet.add((int) (Math.random() * ir.lengthOfTrainingSet()));
        }

        // Train and validate network
        boolean train = true;
        int lastNumberOfRightAnswers = 0;
        while (train) {

            Date trainingsRoundStarted = Calendar.getInstance().getTime();

            // Train
            // Starting up 4 threads (4-core CPU)
            // Still ugly as no loops are used to built and start the threads

            Thread t0 = new Thread(new BackPropagateThread(0, (ir.lengthOfTrainingSet() / 4) * 1, learningRate, nn));
            Thread t1 = new Thread(new BackPropagateThread((ir.lengthOfTrainingSet() / 4) * 1 + 1, (ir.lengthOfTrainingSet() / 4) * 2, learningRate, nn));
            Thread t2 = new Thread(new BackPropagateThread((ir.lengthOfTrainingSet() / 4) * 2 + 1, (ir.lengthOfTrainingSet() / 4) * 3, learningRate, nn));
            Thread t3 = new Thread(new BackPropagateThread((ir.lengthOfTrainingSet() / 4) * 3 + 1, (ir.lengthOfTrainingSet() / 4) * 4, learningRate, nn));

            threadMessage("initiating start to 4 threads");
            t0.start();
            t1.start();
            t2.start();
            t3.start();

            try {
                threadMessage("waiting to continue for " + t0.getName());
                t0.join();
                Thread.sleep(100); // After including these sleeps I din't get some nullpointer exceptions
                threadMessage("waiting to continue for " + t1.getName());
                t1.join();
                Thread.sleep(100);
                threadMessage("waiting to continue for " + t2.getName());
                t2.join();
                Thread.sleep(100);
                threadMessage("waiting to continue for " + t3.getName());
                t3.join();

            } catch (Exception e) {
                System.err.println("Something went wrong");
                System.err.println(e.toString());
            }

//                        Date trainingsRoundStarted = Calendar.getInstance().getTime();
//                        for (int i = 1; i <= ir.lengthOfTrainingSet(); i++) {
//                                double[] image = ir.readImage(i, ir.getTrainingSet());
//                                int label = ir.readLabel(i, ir.getTrainingSet());
//                                double[] actualOutput = nn.forwardPropagate(image);
//                                nn.backPropagate(actualOutput, getTargetOutput(label),
//                                        learningRate);
//                        }

            // Validate
            int numberOfRightAnswers = 0;
            for (int i = 0; i < indicesValidationSet.size(); i++) {
                double[] image = ir.readImage(indicesValidationSet.get(i), ir.getTrainingSet());
                int label = ir.readLabel(indicesValidationSet.get(i), ir.getTrainingSet());
                double[] actualOutput = nn.forwardPropagate(image);

                double bestOutput = -100.0;
                int bestIndex = 0;
                for (int j = 0; j < actualOutput.length; j++) {
                    if (actualOutput[j] > bestOutput) {
                        bestOutput = actualOutput[j];
                        bestIndex = j;
                    }
                }

                if (label == bestIndex) {
                    numberOfRightAnswers++;
                }
            }
            Date trainingsRoundEnded = Calendar.getInstance().getTime();
            double percentage = ((double) numberOfRightAnswers / (double) sizeValidationSet) * 100;
            System.out.println("Current success rate: " + numberOfRightAnswers + "/" + sizeValidationSet + " = " + (int) percentage + "%");
            System.out.println("Round time: " + ((trainingsRoundEnded.getTime() - trainingsRoundStarted.getTime()) / 1000) + " seconds");
//                        System.out.println("Current error: " + (sizeValidationSet - numberOfRightAnswers)
//                                + "; Round time: " + ((trainingsRoundEnded.getTime() - trainingsRoundStarted.getTime()) / 1000) + " seconds");

            // Enough trained?					
            if (numberOfRightAnswers < lastNumberOfRightAnswers) {
                train = false;
            } else {
                lastNumberOfRightAnswers = numberOfRightAnswers;
            }
        }

        // Test network
        int numberOfRightAnswers = 0;
        for (int i = 1; i <= ir.lengthOfTestSet(); i++) {
            double[] image = ir.readImage(i, ir.getTestSet());
            int label = ir.readLabel(i, ir.getTestSet());
            double[] actualOutput = nn.forwardPropagate(image);

            double bestOutput = -100.0;
            int bestIndex = 0;
            for (int j = 0; j < actualOutput.length; j++) {
                if (actualOutput[j] > bestOutput) {
                    bestOutput = actualOutput[j];
                    bestIndex = j;
                }
            }

            if (label == bestIndex) {
                numberOfRightAnswers++;
            }
        }

        System.out.println("Found " + numberOfRightAnswers + " right answers in " + ir.lengthOfTestSet() + " tests");
        System.out.println("Accuracy: " + (double) numberOfRightAnswers / (double) ir.lengthOfTestSet() * 100 + "%");

        Date timeFinished = Calendar.getInstance().getTime();
        System.out.println("--- Ended at " + timeFinished + " --- "
                + "Runtime: " + ((timeFinished.getTime() - timeStarted.getTime()) / 1000) + " seconds");
    }

    private static double[] getTargetOutput(int label) {
        double[] targetOutput = new double[10];
        for (int i = 0; i < targetOutput.length; i++) {
            targetOutput[i] = -1.0;
        }
        targetOutput[label] = 1.0;
        return targetOutput;
    }

    private static void printNetwork(NeuralNetwork nn) {
        int layerIndex = 0;
        for (Layer layer : nn.getLayers().subList(2, 3)) {
            System.out.println("Layer " + layerIndex);
            layerIndex++;

            int neuronIndex = 0;
            for (Neuron neuron : layer.getNeurons()) {
                System.out.println("Neuron " + neuronIndex + " Output: "
                        + neuron.getOutput());
                neuronIndex++;

                int connectionIndex = 0;
                for (Connection connection : neuron.getConnections()) {
                    // System.out.println("Connection " + connectionIndex + " Weight: " + connection.getWeight().getValue());
                    connectionIndex++;
                }
            }
        }
    }
}
