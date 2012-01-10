package neuraalnetwork;

import java.util.List;

public class Neuron {
	
	private double output;
	private List<Connection> connections;
	
	public List<Connection> getConnections() {
		return connections;
	}
	
	public void addConnection(Neuron connectionTo, double weight) {
		connections.add(new Connection(connectionTo, weight));
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

}
