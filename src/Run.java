import images.ImageReader;
import neuraalnetwork.Connection;
import neuraalnetwork.Layer;
import neuraalnetwork.NeuralNetwork;
import neuraalnetwork.Neuron;

public class Run {

	public static void main(String[] args) {
		ImageReader ir = new ImageReader();
		
		//Read args
		int numberNeuronsHiddenLayer = 50;
		double learningRate = 0.001;
		
		//Initialize Network		
		NeuralNetwork nn = new NeuralNetwork(numberNeuronsHiddenLayer);
				
		//Train Network
		for (int i = 1; i <= ir.lengthOfTrainingSet(); i++) {
		//for (int i = 1; i < 2; i++) {
			double[] image = ir.readImage(i, ir.getTrainingSet());
			int label = ir.readLabel(i, ir.getTrainingSet());
			double[] actualOutput = nn.forwardPropagate(image);
			nn.backPropagate(actualOutput, getTargetOutput(label), learningRate);
		}
		
		//Test Network
		int numberOfRightAnswers = 0;
		for (int i = 1; i <= ir.lengthOfTestSet(); i++) {
		//for (int i = 1; i < 2; i++) {
			double[] image = ir.readImage(i, ir.getTestSet());
			int label = ir.readLabel(i, ir.getTestSet());
			double[] actualOutput = nn.forwardPropagate(image);
			
			double maxValue = -100.0;
			int bestIndex = 0;
			for (int j = 0; j < actualOutput.length; j++) {
				if (actualOutput[j] > maxValue) {
					maxValue = actualOutput[j];
					bestIndex = j;
				}
			}
			
			System.out.println("Label: " + label + " Found: " + bestIndex);
			if (label == bestIndex) {
				numberOfRightAnswers++;
			}
		}
		
		System.out.println("Found " + numberOfRightAnswers + " right answers in " + ir.lengthOfTestSet() + " tests");
		System.out.println("Accuracy: " + (double) numberOfRightAnswers / (double) ir.lengthOfTestSet() * 100 + "%");
		
		//printNetwork(nn);
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
				System.out.println("Neuron " + neuronIndex + " Output: " + neuron.getOutput());
				neuronIndex++;
				
				int connectionIndex = 0;
				for (Connection connection : neuron.getConnections()) {
					//System.out.println("Connection " + connectionIndex + " Weight: " + connection.getWeight().getValue());
					connectionIndex++;
				}
			}
		}
	}
}
