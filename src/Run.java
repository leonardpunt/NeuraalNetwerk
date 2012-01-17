import images.ImageReader;
import neuraalnetwork.NeuralNetwork;

public class Run {

	public static void main(String[] args) {
		ImageReader ir = new ImageReader();
		
		//Read args
		int numberNeuronsHiddenLayer = 800;
		double learningRate = 0.5;
		
		//Initialize Network
		
		NeuralNetwork nn = new NeuralNetwork(numberNeuronsHiddenLayer);
		
		//Train Network
		for (int i = 1; i <= ir.lengthOfTrainingSet(); i++) {
			double[] image = ir.readImage(i, ir.getTrainingSet());
			double[] actualOutput = nn.forwardPropagate(image);
			nn.backPropagate(actualOutput, getTargetOutput(), learningRate);
			System.out.println(i);
		}
		
		//Test Network
		//for (int i = 1; i <= ir.lengthOfTestSet(); i++) {
		for (int i = 1; i < 2; i++) {
			double[] image = ir.readImage(i, ir.getTestSet());
			double[] actualOutput = nn.forwardPropagate(image);
			for (int j = 0; j < actualOutput.length; j++) {
				System.out.println(actualOutput[j]);
			}
			System.out.println(ir.readLabel(i, ir.getTestSet()));
		}
	}
	
	private static double[] getTargetOutput() {
		double[] targetOutput = new double[10];
		for (int i = 0; i < targetOutput.length; i++) {
			targetOutput[i] = i;
		}
		return targetOutput;
	}
}
