package neuraalnetwork;

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
	public List<Double> backPropagation(List<Double> derivative, double learningRate) {
		
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
