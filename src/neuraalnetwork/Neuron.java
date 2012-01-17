package neuraalnetwork;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
	
	private double output;
	private List<Connection> connections;
	
	public Neuron() {
		connections = new ArrayList<Connection>();
	}
	
	public List<Connection> getConnections() {
		return connections;
	}
	
	public void addConnection(Neuron connectionFrom, Weight weight) {
		connections.add(new Connection(connectionFrom, weight));
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

}
