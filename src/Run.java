
import images.ImageReader;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import neuraalnetwork.Connection;
import neuraalnetwork.Layer;
import neuraalnetwork.NeuralNetwork;
import neuraalnetwork.Neuron;

public class Run {

	public static void main(String[] args) {
		Date timeStarted = Calendar.getInstance().getTime();
		System.out.println("--- Started at " + timeStarted + " ---");
		
		ImageReader ir = new ImageReader();

		/*
		 * RESULTS
		 * 
		 * numberNeuronsHiddenLayer : 10 
		 * 0.008 = 27%; 23%
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
		 */
		
		// Parameters
		int numberNeuronsHiddenLayer = 50;
		double learningRate = 0.008;
		int sizeValidationSet = 50;

		// Initialize network
		NeuralNetwork nn = new NeuralNetwork(numberNeuronsHiddenLayer);

		// Get validation set		
		List<Integer> indicesValidationSet = ir.getIndicesValidationSet(sizeValidationSet);		

		// Train and validate network
		double meanActualOutput = 0.0;
		int counter = 0;
		while (meanActualOutput < 0.8) {	
			counter++;			
			
			// Initializing threads
			int beginIndex = 0;
			int endIndex = 0;
			int numberOfThreads = 4;
			List<Thread> threads = new ArrayList<Thread>();
			for (int i = 1; i <= numberOfThreads; i++) {
				endIndex = (ir.lengthOfTrainingSet() / numberOfThreads) * i;
				Thread t = new Thread(new BackPropagateThread(beginIndex, endIndex, learningRate, nn));
				threads.add(t);
				t.start();
				beginIndex = endIndex + 1;
			}
			
			// Train
			Date trainingsRoundStarted = Calendar.getInstance().getTime();			
            try {
            	for (Thread t : threads) {
            		t.join();
            		Thread.sleep(100); // After including these sleeps I din't get some nullpointer exceptions
            	}
            } catch (Exception e) {
                System.err.println(e.toString());
            }
			Date trainingsRoundEnded = Calendar.getInstance().getTime();
			
			// Validate
			double sumActualOutput = 0.0;
			for (int i = 0; i < indicesValidationSet.size(); i++) {
				double[] image = ir.readImage(indicesValidationSet.get(i), ir.getTrainingSet());
				int label = ir.readLabel(indicesValidationSet.get(i), ir.getTrainingSet());
				double[] actualOutput = nn.forwardPropagate(image);
				sumActualOutput += actualOutput[label];				
			}			
			
			meanActualOutput = sumActualOutput / indicesValidationSet.size();			
			
			System.out.println("Mean actual output: " + meanActualOutput +"; Round: " + counter + "; Time: " 
					+ ((trainingsRoundEnded.getTime() - trainingsRoundStarted.getTime()) / 1000) + " seconds " + Thread.activeCount());
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

			if (label == bestIndex)
				numberOfRightAnswers++;
		}

		System.out.println("Found " + numberOfRightAnswers + " right answers in " + ir.lengthOfTestSet() + " tests");
		System.out.println("Accuracy: " + (double) numberOfRightAnswers	/ (double) ir.lengthOfTestSet() * 100 + "%");
		
		Date timeFinished = Calendar.getInstance().getTime();
		System.out.println("--- Ended at " + timeFinished + " --- " +
				"Runtime: " + ((timeFinished.getTime() - timeStarted.getTime()) / 1000) + " seconds");
	}
    
	private static double[] getTargetOutput(int label) {
		double[] targetOutput = new double[10];
		for (int i = 0; i < targetOutput.length; i++) {
			targetOutput[i] = -1.0;
		}
		targetOutput[label] = 1.0;
		return targetOutput;
	}
	
    private static class BackPropagateThread implements Runnable {

        int beginIndex, endIndex;
        double learningRate;
        NeuralNetwork nn;

        // Constructor for BackPropagateThread since you can't give run() any parameters.
        public BackPropagateThread(int beginIndex, int endIndex, double learningRate, NeuralNetwork nn) {
            this.beginIndex = beginIndex;
            this.endIndex = endIndex;
            this.nn = nn;
            this.learningRate = learningRate;
        }

        public void run() {
        	ImageReader ir = new ImageReader();
            for (int i = this.beginIndex + 1; i < this.endIndex; i++) {
                double[] image = ir.readImage(i, ir.getTrainingSet());
                int label = ir.readLabel(i, ir.getTrainingSet());
                double[] actualOutput = nn.forwardPropagate(image);
                nn.backPropagate(actualOutput, getTargetOutput(label), learningRate);
            }
        }
    }
    
}
