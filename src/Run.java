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
		 * numberNeuronsHiddenLayer : 10 0.008 = 27%; 23%
		 * 
		 * numberNeuronsHiddenLayer : 50 0.05 = 51%; 53% 0.03 = 60%; 57% 0.01 =
		 * 53%; 63%; 59% 0.008 = 57%; 65%; 66%; 63% 0.005 = 68%; 67%; 57%
		 * 
		 * numberNeuronsHiddenLayer : 70 0.008 = 57%; 58
		 * 
		 * numberNeuronsHiddenLayer : 100 0.008 = 40%; 57%; 56%
		 * 
		 * numberNeuronsHiddenLayer : 150 0.008 = 53%
		 */
		int numberNeuronsHiddenLayer = 10;
		double learningRate = 0.008;

		// Initialize network
		NeuralNetwork nn = new NeuralNetwork(numberNeuronsHiddenLayer);

		// Get validation set
		int sizeValidationSet = 50;
		List<Integer> indicesValidationSet = new LinkedList<Integer>();
		for (int i = 1; i <= sizeValidationSet; i++) {
			indicesValidationSet.add((int) (Math.random() * ir
					.lengthOfTrainingSet()));
		}

		// Train and validate network
		boolean train = true;
		int lastNumberOfRightAnswers = 0;
		while (train) {		
			
			// Train
			Date trainingsRoundStarted = Calendar.getInstance().getTime();
			for (int i = 1; i <= ir.lengthOfTrainingSet(); i++) {
				double[] image = ir.readImage(i, ir.getTrainingSet());
				int label = ir.readLabel(i, ir.getTrainingSet());
				double[] actualOutput = nn.forwardPropagate(image);
				nn.backPropagate(actualOutput, getTargetOutput(label),
						learningRate);
			}
			
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
				
				if (label == bestIndex)
					numberOfRightAnswers++;
			}
			
			Date trainingsRoundEnded = Calendar.getInstance().getTime();
			System.out.println("Current error: " + (sizeValidationSet - numberOfRightAnswers) +
					"; Round time: " + ((trainingsRoundEnded.getTime() - trainingsRoundStarted.getTime()) / 1000) + " seconds");
			
			// Enough trained?					
			if (numberOfRightAnswers < lastNumberOfRightAnswers)
				train = false;
			else
				lastNumberOfRightAnswers = numberOfRightAnswers;
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
