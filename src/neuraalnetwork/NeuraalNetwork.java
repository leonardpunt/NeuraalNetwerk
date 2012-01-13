package neuraalnetwork;

import java.util.ArrayList;
import java.util.List;

public class NeuraalNetwork {
	
	private List<Layer> layers;
	
	public void forwardPropagate() {
		// Call each layers forwardPropagate function.
	}
	
	public void backPropagate(double[] actualOutput, double[] desiredOutput, double learningRate) {
		List<List<Double>> derivatives = new ArrayList<List<Double>>();
		
		//Calculate partial derivative for last layer
		List<Double> partialDerivativeLastLayer = new ArrayList<Double>();
		for (int i = 0; i < actualOutput.length; i++) {
			partialDerivativeLastLayer.add(actualOutput[i] - desiredOutput[i]);
		}
		derivatives.add(partialDerivativeLastLayer);
		
		//Iterate through all layers, except the first, starting from the last. 
		//  Call backPropagate function for each layer
		//  The backPropagete function returns the derivatives for that layer, so add that to the derivatives list.
		for (int i = layers.size()-1; i > 1; i--) {
			derivatives.add(0, layers.get(i).backPropagation(derivatives.get(0), learningRate));
		}
	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void setLayers(List<Layer> layers) {
		this.layers = layers;
	}

}
