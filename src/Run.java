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
			
			// Train
			Date trainingsRoundStarted = Calendar.getInstance().getTime();
			for (int i = 1; i <= ir.lengthOfTrainingSet(); i++) {
				double[] image = ir.readImage(i, ir.getTrainingSet());
				int label = ir.readLabel(i, ir.getTrainingSet());
				double[] actualOutput = nn.forwardPropagate(image);
				nn.backPropagate(actualOutput, getTargetOutput(label),
						learningRate);
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
					+ ((trainingsRoundEnded.getTime() - trainingsRoundStarted.getTime()) / 1000) + " seconds");
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

//	private static void printNetwork(NeuralNetwork nn) {
//		int layerIndex = 0;
//		for (Layer layer : nn.getLayers().subList(2, 3)) {
//			System.out.println("Layer " + layerIndex);
//			layerIndex++;
//
//			int neuronIndex = 0;
//			for (Neuron neuron : layer.getNeurons()) {
//				System.out.println("Neuron " + neuronIndex + " Output: "
//						+ neuron.getOutput());
//				neuronIndex++;
//
//				int connectionIndex = 0;
//				for (Connection connection : neuron.getConnections()) {
//					// System.out.println("Connection " + connectionIndex + " Weight: " + connection.getWeight().getValue());
//					connectionIndex++;
//				}
//			}
//		}
//	}
	
//	double blabla = 0.0;
//	int testLabel = ir.readLabel(1, ir.getTrainingSet());
//	System.out.println(testLabel);
//	double[] testImage = ir.readImage(1, ir.getTrainingSet());
//	double[] testOutput = nn.forwardPropagate(testImage);
//	System.out.println(testOutput[testLabel]);
//	int counter = 0;
//	
//	while (blabla < 0.90) {		
//		counter++;
//		for (int i = 1; i <= 1; i++) {
//			double[] image = ir.readImage(i, ir.getTrainingSet());
//			int label = ir.readLabel(i, ir.getTrainingSet());
//			double[] actualOutput = nn.forwardPropagate(image);
//			nn.backPropagate(actualOutput, getTargetOutput(label),
//					learningRate);
//		}				
//		testOutput = nn.forwardPropagate(testImage);
//		System.out.println(counter + " : " + testOutput[testLabel]);
//		
//		blabla = testOutput[testLabel];
//	}
//	
//	System.out.println("-----------------------");
//	
//	for (int i = 0; i < 10; i++) {
//		System.out.println(testOutput[i]);
//	}
//	
//	System.exit(-1);
}
