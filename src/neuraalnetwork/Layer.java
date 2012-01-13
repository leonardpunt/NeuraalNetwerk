package neuraalnetwork;

import java.util.ArrayList;
import java.util.List;

import activation.HyperbolicTanget;

public class Layer {
	
	private Layer prevLayer;
	private List<Neuron> neurons;
	
	public void forwardPropagate() {
		if (prevLayer != null) {			
			for (Neuron neuron : neurons) {
				double sum = 0.0;
				for (Neuron neuronPrevLayer : prevLayer.getNeurons()) {				
					for (Connection connection : neuronPrevLayer.getConnections()) {
						if (connection.getConnectionTo().equals(neuron)) {
							sum += neuronPrevLayer.getOutput() * connection.getWeight();
						}
					}
				}
			neuron.setOutput(HyperbolicTanget.activate(sum));
			}
		}
		else {
			//Input laag
		}
		
	}
	
	//Returns derivative
	public List<Double> backPropagation(List<Double> derivativesX, double learningRate) {
		//Step 1: equation 3 (y)
		//Step 2: equation 4 (w)
		//Step 3: equation 5 (xN-1)
		//Step 4: equation 6 and update the weights
		
		List<Double> derivativesY = new ArrayList<Double>();
		List<Double> derivativesW = new ArrayList<Double>();
		List<Double> derivativesXPreviousLayer = new ArrayList<Double>();
		
		//Calculate derivatives for y
		for (int i = 0; i < neurons.size(); i++) {
			double output = neurons.get(i).getOutput();
			derivativesY.add(HyperbolicTanget.derivative(output) * derivativesX.get(i));
		}

		//KLOPT NIET
		//Calculate derivatives for w
		for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++) {			
			for(int connectionIndex = 0; connectionIndex < neurons.get(neuronIndex).getConnections().size(); connectionIndex++) {
				double output = prevLayer.getNeurons().get(neuronIndex).getOutput();				
				derivativesW.add(derivativesY.get(neuronIndex) * output);				
			}
		}
		
		//NOG NIET AF
		//Calculate derivatives for x, for the previous layer
		for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++) {
			double sum = 0.0;
			for(int connectionIndex = 0; connectionIndex < neurons.get(neuronIndex).getConnections().size(); connectionIndex++) {
				sum += derivativesY.get(neuronIndex); 
			}
			derivativesXPreviousLayer.add(sum);
		}
		
		
		
		return null;		
	}

	public Layer getPrevLayer() {
		return prevLayer;
	}

	public void setPrevLayer(Layer prevLayer) {
		this.prevLayer = prevLayer;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeurons(List<Neuron> neurons) {
		this.neurons = neurons;
	}

}
