package neuraalnetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import activation.HyperbolicTanget;

public class Layer {
	
	private Layer prevLayer;
	private List<Neuron> neurons;
	
	public Layer(Layer prevLayer) {
		this.prevLayer = prevLayer;
	}
	
	public void forwardPropagate() {
		if (prevLayer != null) {		
			for (Neuron neuron : neurons) {
				double sum = 0.0;
				for (Connection connection : neuron.getConnections()) {
					sum += connection.getWeight().getValue() * connection.getConnectionFrom().getOutput();
				}
				neuron.setOutput(HyperbolicTanget.activate(sum));
			}
		}
	}
	
	//Returns derivative for previous layer
	public Map<Neuron, Double> backPropagation(Map<Neuron, Double> derivativesX, double learningRate) {		
		Map<Neuron, Double> derivativesY = new HashMap<Neuron, Double>();
		Map<Weight, Double> derivativesW = new HashMap<Weight, Double>();
		Map<Neuron, Double> derivativesXPreviousLayer = new HashMap<Neuron, Double>();
		
		//Calculate derivatives for y		
		for (Neuron neuron : neurons) {
			double output = neuron.getOutput();
			derivativesY.put(neuron, HyperbolicTanget.derivative(output) * derivativesX.get(neuron));
		}

		//Calculate derivatives for w
		for (Neuron neuron : neurons) {
			for (Connection connection : neuron.getConnections()) {					
				double output = connection.getConnectionFrom().getOutput();				
				derivativesW.put(connection.getWeight(), output * derivativesY.get(neuron));				
			}
		}
		
		//Calculate derivatives for x, for the previous layer
		for (Neuron neuron : neurons) {
			for (Connection connection : neuron.getConnections()) {
				derivativesXPreviousLayer.put(connection.getConnectionFrom(), 
						derivativesY.get(neuron) * derivativesW.get(connection.getWeight()));
			}
		}
		
		//Update the weights in this layer
		for (Neuron neuron : neurons) {
			for (Connection connection : neuron.getConnections()) {
				double oldValue = connection.getWeight().getValue();
				double newValue = oldValue - learningRate * derivativesW.get(connection.getWeight());
				connection.getWeight().setValue(newValue);
			}
		}
		
		return derivativesXPreviousLayer;		
	}

	public Layer getPrevLayer() {
		return prevLayer;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeurons(List<Neuron> neurons) {
		this.neurons = neurons;
	}

}
