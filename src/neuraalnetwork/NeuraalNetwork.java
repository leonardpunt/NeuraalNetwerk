package neuraalnetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeuraalNetwork {
	
	private List<Layer> layers;
	
	public void forwardPropagate() {
		// Call each layers forwardPropagate function.
	}
	
	public void backPropagate(double[] actualOutput, double[] desiredOutput, double learningRate) {
		List<Map<Neuron, Double>> layerDerivativesX = new ArrayList<Map<Neuron, Double>>();
		
		//Calculate derivatives for x, for last layer
		Map<Neuron, Double> derivativesX = new HashMap<Neuron, Double>();
		for (int i = 0; i < actualOutput.length; i++) {
			Neuron neuron = layers.get(layers.size()).getNeurons().get(i);
			derivativesX.put(neuron, actualOutput[i] - desiredOutput[i]);
		}
		layerDerivativesX.add(derivativesX);
		
		//Iterate through all layers, except the first, starting from the last. 
		//  - Call backPropagate function for each layer
		//  - The backPropagete function returns the derivativesX for that layer, so add that to the derivatives list.
		for (int i = layers.size()-1; i > 1; i--) {
			layerDerivativesX.add(0, layers.get(i).backPropagation(layerDerivativesX.get(0), learningRate));
		}
	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void setLayers(List<Layer> layers) {
		this.layers = layers;
	}

}
