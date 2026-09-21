"""CryoNet Interactive — constrained cryogenic feed-system network solver.

Portfolio project by Janvi Sharma.

Install once:
    pip install numpy scipy matplotlib

Run the interactive dashboard:
    python CryoNet_Interactive.py

Export static PNG and CSV results:
    python CryoNet_Interactive.py --export

This single file contains LOX and liquid-methane property models, reusable
hydraulic components, two independent nonlinear network solves, operating-
envelope checks, oxidizer/fuel mixture-ratio analysis, CSV reporting, plots,
and an interactive browser dashboard.

Design basis:
  * Reference engine: NASA Project Morpheus HD4-A baseline, a pressure-fed
    LOX/methane lander engine in the 4,200 lbf class.
  * Published program requirements: 4,200 lbf thrust, 215 s specific impulse,
    4:1 throttling, and at least 210 s run time (NASA NTRS 20110014012).
  * Published Morpheus-family injector design bands: approximately 25-32% of
    chamber pressure at full thrust and 9-11% at 4:1 throttle; testing measured
    roughly 15% at low throttle (NASA NTRS 20140009917).
  * A 325 psia minimum engine inlet and 350 psia minimum tank operating point
    are taken from the NASA-hardware-based Liquid Methane Propulsion System
    Testbed thesis. The 250 psia absolute chamber design point is derived from
    325 psia / 1.30 so the nominal injector drop is 30% of chamber pressure.
  * Total design flow is calculated from thrust/(Isp*g0). Global O/F=3.40,
    liquid temperatures, equivalent line resistance, and Cd=0.75 are explicitly
    identified engineering assumptions because exact flight plumbing and
    injector dimensions are not public.
  * The node/branch formulation follows the same conservation structure used
    by NASA GFSSP: mass conservation at nodes and a pressure relation on every
    branch, solved simultaneously.

It is a steady, one-dimensional, single-phase educational model—not flight
software, an exact Morpheus replica, a combustion/transient model, or a
substitute for validated property data and hardware testing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import threading
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import exp, log10, pi
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import least_squares



# Fluid properties



@dataclass(frozen=True)
class Fluid:
    """Lightweight liquid-property model around a reference temperature."""

    name: str
    reference_temperature_k: float
    reference_density_kg_m3: float
    reference_viscosity_pa_s: float
    specific_heat_j_kg_k: float
    thermal_expansion_1_k: float
    viscosity_temperature_coefficient_1_k: float
    normal_boiling_temperature_k: float
    normal_boiling_pressure_pa: float
    latent_heat_j_kg: float
    vapor_gas_constant_j_kg_k: float
    valid_temperature_range_k: tuple[float, float]

    def _validate_temperature(self, temperature_k: float) -> None:
        low, high = self.valid_temperature_range_k
        if not low <= temperature_k <= high:
            raise ValueError(
                f"{self.name} model is valid from {low:.1f} to {high:.1f} K; "
                f"received {temperature_k:.2f} K."
            )

    def density(self, temperature_k: float) -> float:
        self._validate_temperature(temperature_k)
        delta_t = temperature_k - self.reference_temperature_k
        value = self.reference_density_kg_m3 * (
            1.0 - self.thermal_expansion_1_k * delta_t
        )
        if value <= 0.0:
            raise ValueError("Density correlation produced a nonphysical value.")
        return value

    def viscosity(self, temperature_k: float) -> float:
        self._validate_temperature(temperature_k)
        delta_t = temperature_k - self.reference_temperature_k
        return self.reference_viscosity_pa_s * exp(
            -self.viscosity_temperature_coefficient_1_k * delta_t
        )

    def specific_heat(self, temperature_k: float) -> float:
        self._validate_temperature(temperature_k)
        return self.specific_heat_j_kg_k

    def vapor_pressure(self, temperature_k: float) -> float:
        """Approximate saturation pressure using Clausius-Clapeyron."""
        self._validate_temperature(temperature_k)
        exponent = -(self.latent_heat_j_kg / self.vapor_gas_constant_j_kg_k) * (
            1.0 / temperature_k - 1.0 / self.normal_boiling_temperature_k
        )
        return self.normal_boiling_pressure_pa * exp(exponent)


LOX = Fluid(
    name="Liquid oxygen (LOX)",
    reference_temperature_k=90.0,
    reference_density_kg_m3=1142.0,
    reference_viscosity_pa_s=1.95e-4,
    specific_heat_j_kg_k=1700.0,
    thermal_expansion_1_k=4.4e-3,
    viscosity_temperature_coefficient_1_k=2.5e-2,
    normal_boiling_temperature_k=90.188,
    normal_boiling_pressure_pa=101_325.0,
    latent_heat_j_kg=213_000.0,
    vapor_gas_constant_j_kg_k=259.8,
    valid_temperature_range_k=(80.0, 115.0),
)


# Approximate methane properties


LCH4 = Fluid(
    name="Liquid methane (LCH4)",
    reference_temperature_k=112.0,
    reference_density_kg_m3=422.6,
    reference_viscosity_pa_s=1.17e-4,
    specific_heat_j_kg_k=3_500.0,
    thermal_expansion_1_k=3.6e-3,
    viscosity_temperature_coefficient_1_k=2.2e-2,
    normal_boiling_temperature_k=111.66,
    normal_boiling_pressure_pa=101_325.0,
    latent_heat_j_kg=510_000.0,
    vapor_gas_constant_j_kg_k=518.3,
    valid_temperature_range_k=(95.0, 135.0),
)



# Hydraulic components



GRAVITY_M_S2 = 9.80665
PSI_TO_PA = 6_894.757293168


def flow_quantities(
    mass_flow_kg_s: float,
    diameter_m: float,
    fluid: Fluid,
    temperature_k: float,
) -> tuple[float, float, float]:
    """Return velocity, Reynolds number, and dynamic pressure."""
    density = fluid.density(temperature_k)
    viscosity = fluid.viscosity(temperature_k)
    area = pi * diameter_m**2 / 4.0
    velocity = mass_flow_kg_s / (density * area)
    reynolds = density * abs(velocity) * diameter_m / viscosity
    dynamic_pressure = 0.5 * density * velocity**2
    return velocity, reynolds, dynamic_pressure


def darcy_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Darcy friction factor: laminar theory or turbulent Haaland equation."""
    if reynolds <= 0.0:
        return 0.0
    if reynolds < 2300.0:
        return 64.0 / reynolds
    inverse_sqrt_f = -1.8 * log10(
        (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds
    )
    return 1.0 / inverse_sqrt_f**2


class Component(ABC):
    """Base class for one hydraulic branch component."""

    @abstractmethod
    def pressure_loss_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        pass

    def pressure_gain_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        return 0.0

    def heat_to_fluid_w(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        return 0.0

    def metrics(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> dict[str, float]:
        return {}


@dataclass
class Pipe(Component):
    length_m: float
    diameter_m: float
    roughness_m: float = 1.5e-6
    minor_loss_k: float = 0.0
    elevation_change_m: float = 0.0

    def pressure_loss_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        velocity, reynolds, dynamic_pressure = flow_quantities(
            mass_flow_kg_s, self.diameter_m, fluid, temperature_k
        )
        del velocity
        friction = darcy_friction_factor(
            reynolds, self.roughness_m / self.diameter_m
        )
        friction_loss = (
            friction * self.length_m / self.diameter_m + self.minor_loss_k
        ) * dynamic_pressure
        static_head = (
            fluid.density(temperature_k) * GRAVITY_M_S2 * self.elevation_change_m
        )
        return friction_loss + static_head

    def metrics(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> dict[str, float]:
        velocity, reynolds, _ = flow_quantities(
            mass_flow_kg_s, self.diameter_m, fluid, temperature_k
        )
        friction = darcy_friction_factor(
            reynolds, self.roughness_m / self.diameter_m
        )
        return {
            "velocity_m_s": velocity,
            "reynolds": reynolds,
            "friction_factor": friction,
        }


@dataclass
class Valve(Component):
    diameter_m: float
    full_open_loss_k: float
    opening_fraction: float = 1.0

    @property
    def effective_loss_k(self) -> float:
        # Approximate valve loss
        return self.full_open_loss_k / self.opening_fraction**2

    def pressure_loss_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        _, _, dynamic_pressure = flow_quantities(
            mass_flow_kg_s, self.diameter_m, fluid, temperature_k
        )
        return self.effective_loss_k * dynamic_pressure

    def metrics(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> dict[str, float]:
        velocity, reynolds, _ = flow_quantities(
            mass_flow_kg_s, self.diameter_m, fluid, temperature_k
        )
        return {
            "velocity_m_s": velocity,
            "reynolds": reynolds,
            "loss_coefficient_k": self.effective_loss_k,
        }


@dataclass
class InjectorOrifice(Component):
    """Equivalent total injector flow area using the liquid orifice equation.

    ``flow_area_m2`` is the sum of all active injector-element areas, not the
    diameter of one hole. The discharge coefficient is deliberately explicit so
    measured cold-flow data can replace the preliminary value later.
    """

    flow_area_m2: float
    discharge_coefficient: float = 0.75

    def pressure_loss_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        density = fluid.density(temperature_k)
        effective_area = self.discharge_coefficient * self.flow_area_m2
        return (mass_flow_kg_s / effective_area) ** 2 / (2.0 * density)

    def metrics(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> dict[str, float]:
        density = fluid.density(temperature_k)
        velocity = mass_flow_kg_s / (density * self.flow_area_m2)
        return {
            "total_flow_area_mm2": self.flow_area_m2 * 1.0e6,
            "discharge_coefficient": self.discharge_coefficient,
            "bulk_velocity_m_s": velocity,
        }


@dataclass
class Pump(Component):
    shutoff_pressure_rise_pa: float
    curve_coefficient_pa_per_kg2_s2: float
    efficiency: float = 0.68
    required_npsh_m: float = 2.5

    def pressure_loss_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        return 0.0

    def pressure_gain_pa(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        return self.shutoff_pressure_rise_pa - (
            self.curve_coefficient_pa_per_kg2_s2 * mass_flow_kg_s**2
        )

    def heat_to_fluid_w(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> float:
        density = fluid.density(temperature_k)
        hydraulic_power = max(
            0.0,
            self.pressure_gain_pa(mass_flow_kg_s, fluid, temperature_k)
            * mass_flow_kg_s
            / density,
        )
        return hydraulic_power * (1.0 / self.efficiency - 1.0)

    def metrics(
        self, mass_flow_kg_s: float, fluid: Fluid, temperature_k: float
    ) -> dict[str, float]:
        gain = self.pressure_gain_pa(mass_flow_kg_s, fluid, temperature_k)
        density = fluid.density(temperature_k)
        hydraulic_power = max(0.0, gain * mass_flow_kg_s / density)
        return {
            "pressure_gain_pa": gain,
            "hydraulic_power_w": hydraulic_power,
            "shaft_power_w": hydraulic_power / self.efficiency,
            "required_npsh_m": self.required_npsh_m,
        }



# Network solver



@dataclass
class Node:
    name: str
    fixed_pressure_pa: float | None = None
    temperature_k: float | None = None
    initial_pressure_pa: float | None = None


@dataclass
class Edge:
    name: str
    start: str
    end: str
    component: Component
    heat_leak_w: float = 0.0


@dataclass
class EdgeResult:
    name: str
    start: str
    end: str
    mass_flow_kg_s: float
    pressure_loss_pa: float
    pressure_gain_pa: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Solution:
    fluid: Fluid
    node_pressures_pa: dict[str, float]
    edge_results: dict[str, EdgeResult]
    converged: bool
    residual_norm: float
    reference_temperature_k: float
    node_temperatures_k: dict[str, float] = field(default_factory=dict)
    edge_outlet_temperatures_k: dict[str, float] = field(default_factory=dict)

    def flow_into(self, node_name: str) -> float:
        return sum(
            item.mass_flow_kg_s
            for item in self.edge_results.values()
            if item.end == node_name
        )

    def saturation_margin_pa(self, node_name: str) -> float:
        temperature = self.node_temperatures_k.get(
            node_name, self.reference_temperature_k
        )
        return self.node_pressures_pa[node_name] - self.fluid.vapor_pressure(temperature)

    def npsh_available_m(self, node_name: str) -> float:
        temperature = self.node_temperatures_k.get(
            node_name, self.reference_temperature_k
        )
        return self.saturation_margin_pa(node_name) / (
            self.fluid.density(temperature) * GRAVITY_M_S2
        )


class Network:
    """Directed steady-state node-pressure / branch-flow network."""

    def __init__(self, name: str, fluid: Fluid, reference_temperature_k: float):
        self.name = name
        self.fluid = fluid
        self.reference_temperature_k = reference_temperature_k
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Duplicate node: {node.name}")
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.start not in self.nodes or edge.end not in self.nodes:
            raise ValueError(f"Unknown node on edge {edge.name}")
        if any(existing.name == edge.name for existing in self.edges):
            raise ValueError(f"Duplicate edge: {edge.name}")
        self.edges.append(edge)

    def solve(self, initial_mass_flow_kg_s: float = 2.5) -> Solution:
        free_nodes = [
            node for node in self.nodes.values() if node.fixed_pressure_pa is None
        ]
        free_index = {node.name: index for index, node in enumerate(free_nodes)}
        number_of_pressures = len(free_nodes)
        fixed_pressures = [
            node.fixed_pressure_pa
            for node in self.nodes.values()
            if node.fixed_pressure_pa is not None
        ]
        if len(fixed_pressures) < 2:
            raise ValueError("The network needs two fixed-pressure boundaries.")

        default_pressure = float(np.mean(fixed_pressures))
        pressure_guesses = [
            node.initial_pressure_pa or default_pressure for node in free_nodes
        ]
        flow_guesses = [initial_mass_flow_kg_s] * len(self.edges)
        x0 = np.array(pressure_guesses + flow_guesses, dtype=float)
        lower = np.array(
            [1_000.0] * number_of_pressures + [1.0e-9] * len(self.edges)
        )
        upper = np.array(
            [50.0e6] * number_of_pressures + [100.0] * len(self.edges)
        )

        def node_pressure(name: str, vector: np.ndarray) -> float:
            node = self.nodes[name]
            if node.fixed_pressure_pa is not None:
                return node.fixed_pressure_pa
            return float(vector[free_index[name]])

        def residual(vector: np.ndarray) -> np.ndarray:
            flows = vector[number_of_pressures:]
            equations: list[float] = []

            # Pressure equations
            for edge, mass_flow in zip(self.edges, flows, strict=True):
                pressure_start = node_pressure(edge.start, vector)
                pressure_end = node_pressure(edge.end, vector)
                loss = edge.component.pressure_loss_pa(
                    mass_flow, self.fluid, self.reference_temperature_k
                )
                gain = edge.component.pressure_gain_pa(
                    mass_flow, self.fluid, self.reference_temperature_k
                )
                equations.append(
                    (pressure_start - pressure_end + gain - loss) / 100_000.0
                )

            # Mass conservation
            for node in free_nodes:
                inflow = sum(
                    flow
                    for edge, flow in zip(self.edges, flows, strict=True)
                    if edge.end == node.name
                )
                outflow = sum(
                    flow
                    for edge, flow in zip(self.edges, flows, strict=True)
                    if edge.start == node.name
                )
                equations.append(inflow - outflow)
            return np.asarray(equations)

        numerical_result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=5_000,
        )
        pressures = {
            name: (
                node.fixed_pressure_pa
                if node.fixed_pressure_pa is not None
                else float(numerical_result.x[free_index[name]])
            )
            for name, node in self.nodes.items()
        }
        flows = numerical_result.x[number_of_pressures:]
        edge_results: dict[str, EdgeResult] = {}
        for edge, flow in zip(self.edges, flows, strict=True):
            edge_results[edge.name] = EdgeResult(
                name=edge.name,
                start=edge.start,
                end=edge.end,
                mass_flow_kg_s=float(flow),
                pressure_loss_pa=edge.component.pressure_loss_pa(
                    flow, self.fluid, self.reference_temperature_k
                ),
                pressure_gain_pa=edge.component.pressure_gain_pa(
                    flow, self.fluid, self.reference_temperature_k
                ),
                metrics=edge.component.metrics(
                    flow, self.fluid, self.reference_temperature_k
                ),
            )

        solution = Solution(
            fluid=self.fluid,
            node_pressures_pa=pressures,
            edge_results=edge_results,
            converged=bool(
                numerical_result.success
                and np.linalg.norm(numerical_result.fun) < 1.0e-6
            ),
            residual_norm=float(np.linalg.norm(numerical_result.fun)),
            reference_temperature_k=self.reference_temperature_k,
        )
        self._solve_temperatures(solution)
        return solution

    def _solve_temperatures(self, solution: Solution) -> None:
        """Post-process sensible heat for this directed acyclic network."""
        incoming = {name: [] for name in self.nodes}
        outgoing = {name: [] for name in self.nodes}
        indegree = {name: 0 for name in self.nodes}
        for edge in self.edges:
            incoming[edge.end].append(edge)
            outgoing[edge.start].append(edge)
            indegree[edge.end] += 1

        queue = [name for name, degree in indegree.items() if degree == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for edge in outgoing[current]:
                indegree[edge.end] -= 1
                if indegree[edge.end] == 0:
                    queue.append(edge.end)
        if len(order) != len(self.nodes):
            return

        node_temperatures: dict[str, float] = {}
        outlet_temperatures: dict[str, float] = {}
        for node_name in order:
            node = self.nodes[node_name]
            if not incoming[node_name] and node.temperature_k is not None:
                temperature = node.temperature_k
            elif incoming[node_name]:
                weighted_temperature = 0.0
                total_flow = 0.0
                for edge in incoming[node_name]:
                    flow = solution.edge_results[edge.name].mass_flow_kg_s
                    weighted_temperature += flow * outlet_temperatures[edge.name]
                    total_flow += flow
                temperature = weighted_temperature / total_flow
            else:
                temperature = self.reference_temperature_k
            node_temperatures[node_name] = temperature

            for edge in outgoing[node_name]:
                flow = solution.edge_results[edge.name].mass_flow_kg_s
                heat = edge.heat_leak_w + edge.component.heat_to_fluid_w(
                    flow, self.fluid, temperature
                )
                outlet_temperatures[edge.name] = temperature + heat / (
                    flow * self.fluid.specific_heat(temperature)
                )

        solution.node_temperatures_k = node_temperatures
        solution.edge_outlet_temperatures_k = outlet_temperatures

    def pump_check(self, solution: Solution) -> dict[str, float | bool]:
        for edge in self.edges:
            if isinstance(edge.component, Pump):
                available = solution.npsh_available_m(edge.start)
                required = edge.component.required_npsh_m
                return {
                    "available_m": available,
                    "required_m": required,
                    "margin_m": available - required,
                    "passes": available >= required,
                }
        raise ValueError("No pump exists in the network.")



# LOX transfer system



@dataclass(frozen=True)
class Scenario:
    name: str
    main_valve_opening: float = 1.0
    bypass_valve_opening: float = 0.35
    pump_health: float = 1.0
    source_pressure_pa: float = 450_000.0
    source_temperature_k: float = 90.0
    receiver_pressure_pa: float = 300_000.0


NODE_POSITIONS = {
    "Storage tank": (0.0, 0.0),
    "Pump inlet": (1.5, 0.0),
    "Pump outlet": (3.0, 0.0),
    "Main branch": (4.5, 1.2),
    "Bypass branch": (4.5, -1.2),
    "Branch merge": (6.2, 0.0),
    "Coupler outlet": (7.8, 0.0),
    "Vehicle tank": (9.4, 0.0),
}


def build_transfer_network(scenario: Scenario) -> Network:
    network = Network(
        f"LOX transfer — {scenario.name}", LOX, scenario.source_temperature_k
    )
    network.add_node(
        Node(
            "Storage tank",
            fixed_pressure_pa=scenario.source_pressure_pa,
            temperature_k=scenario.source_temperature_k,
        )
    )
    network.add_node(Node("Pump inlet", initial_pressure_pa=420_000.0))
    network.add_node(Node("Pump outlet", initial_pressure_pa=1_000_000.0))
    network.add_node(Node("Main branch", initial_pressure_pa=900_000.0))
    network.add_node(Node("Bypass branch", initial_pressure_pa=850_000.0))
    network.add_node(Node("Branch merge", initial_pressure_pa=750_000.0))
    network.add_node(Node("Coupler outlet", initial_pressure_pa=500_000.0))
    network.add_node(
        Node("Vehicle tank", fixed_pressure_pa=scenario.receiver_pressure_pa)
    )

    network.add_edge(
        Edge(
            "Suction line",
            "Storage tank",
            "Pump inlet",
            Pipe(2.0, 0.032, minor_loss_k=1.5),
            heat_leak_w=12.0,
        )
    )
    network.add_edge(
        Edge(
            "Transfer pump",
            "Pump inlet",
            "Pump outlet",
            Pump(700_000.0 * scenario.pump_health, 18_000.0),
        )
    )
    network.add_edge(
        Edge(
            "Main valve",
            "Pump outlet",
            "Main branch",
            Valve(0.028, 2.5, scenario.main_valve_opening),
        )
    )
    network.add_edge(
        Edge(
            "Main transfer line",
            "Main branch",
            "Branch merge",
            Pipe(5.0, 0.028, minor_loss_k=3.0),
            heat_leak_w=35.0,
        )
    )
    network.add_edge(
        Edge(
            "Bypass valve",
            "Pump outlet",
            "Bypass branch",
            Valve(0.018, 8.0, scenario.bypass_valve_opening),
        )
    )
    network.add_edge(
        Edge(
            "Bypass line",
            "Bypass branch",
            "Branch merge",
            Pipe(6.0, 0.018, minor_loss_k=4.0),
            heat_leak_w=25.0,
        )
    )
    network.add_edge(
        Edge(
            "Cryogenic coupler",
            "Branch merge",
            "Coupler outlet",
            Valve(0.028, 10.0),
            heat_leak_w=6.0,
        )
    )
    network.add_edge(
        Edge(
            "Vehicle fill line",
            "Coupler outlet",
            "Vehicle tank",
            Pipe(2.5, 0.028, minor_loss_k=5.0),
            heat_leak_w=18.0,
        )
    )
    return network


def solve_scenario(scenario: Scenario) -> tuple[Network, Solution]:
    network = build_transfer_network(scenario)
    solution = network.solve()
    if not solution.converged:
        raise RuntimeError(
            f"{scenario.name} did not converge; residual={solution.residual_norm:.3e}"
        )
    return network, solution



# Dual-propellant feed system



@dataclass(frozen=True)
class EngineDesignBasis:
    """Traceable requirements and explicitly identified analysis assumptions."""

    engine_name: str = "NASA Project Morpheus HD4-A baseline"
    architecture: str = "pressure-fed LOX/LCH4"
    design_thrust_n: float = 4_200.0 * 4.4482216152605
    specific_impulse_s: float = 215.0
    throttle_ratio: float = 4.0
    minimum_run_time_s: float = 210.0
    design_tank_pressure_pa: float = 350.0 * PSI_TO_PA
    minimum_engine_inlet_pressure_pa: float = 325.0 * PSI_TO_PA
    design_chamber_pressure_pa: float = 250.0 * PSI_TO_PA
    target_mixture_ratio: float = 3.40
    mixture_ratio_tolerance: float = 0.25
    injector_ratio_target: float = 0.30
    full_thrust_injector_band: tuple[float, float] = (0.25, 0.32)
    low_throttle_injector_band: tuple[float, float] = (0.09, 0.18)
    design_valve_opening: float = 0.82
    common_valve_mismatch_limit: float = 0.02
    lox_operating_temperature_k: tuple[float, float] = (87.0, 95.0)
    methane_operating_temperature_k: tuple[float, float] = (108.0, 120.0)

    @property
    def design_total_flow_kg_s(self) -> float:
        return self.design_thrust_n / (self.specific_impulse_s * GRAVITY_M_S2)

    @property
    def lox_design_flow_kg_s(self) -> float:
        ratio = self.target_mixture_ratio
        return self.design_total_flow_kg_s * ratio / (1.0 + ratio)

    @property
    def methane_design_flow_kg_s(self) -> float:
        return self.design_total_flow_kg_s / (1.0 + self.target_mixture_ratio)

    def injector_band(self, chamber_pressure_pa: float) -> tuple[float, float]:
        """Interpolate the published low- and full-thrust Morpheus bands."""
        power = chamber_pressure_pa / self.design_chamber_pressure_pa
        blend = min(1.0, max(0.0, (power - 0.25) / 0.75))
        low = self.low_throttle_injector_band
        high = self.full_thrust_injector_band
        return (
            low[0] + blend * (high[0] - low[0]),
            low[1] + blend * (high[1] - low[1]),
        )


MORPHEUS = EngineDesignBasis()


def _injector_area_from_design(
    design_flow_kg_s: float,
    fluid: Fluid,
    temperature_k: float,
    discharge_coefficient: float,
) -> float:
    pressure_drop = (
        MORPHEUS.injector_ratio_target * MORPHEUS.design_chamber_pressure_pa
    )
    return design_flow_kg_s / (
        discharge_coefficient
        * (2.0 * fluid.density(temperature_k) * pressure_drop) ** 0.5
    )


@dataclass(frozen=True)
class FeedHardware:
    """Equivalent one-dimensional hardware for one propellant circuit."""

    label: str
    fluid: Fluid
    design_mass_flow_kg_s: float
    line_diameter_m: float
    upstream_length_m: float
    downstream_length_m: float
    upstream_minor_loss_k: float
    downstream_minor_loss_k: float
    injector_total_area_m2: float
    injector_discharge_coefficient: float
    nominal_temperature_k: float
    heat_leak_w: float
    color: str


LOX_FEED = FeedHardware(
    label="LOX",
    fluid=LOX,
    design_mass_flow_kg_s=MORPHEUS.lox_design_flow_kg_s,
    line_diameter_m=0.040,
    upstream_length_m=1.2,
    downstream_length_m=1.6,
    upstream_minor_loss_k=1.5,
    downstream_minor_loss_k=2.5,
    injector_total_area_m2=_injector_area_from_design(
        MORPHEUS.lox_design_flow_kg_s, LOX, 90.0, 0.75
    ),
    injector_discharge_coefficient=0.75,
    nominal_temperature_k=90.0,
    heat_leak_w=22.0,
    color="#53e4ff",
)

METHANE_FEED = FeedHardware(
    label="LCH4",
    fluid=LCH4,
    design_mass_flow_kg_s=MORPHEUS.methane_design_flow_kg_s,
    line_diameter_m=0.032,
    upstream_length_m=1.2,
    downstream_length_m=1.6,
    upstream_minor_loss_k=1.5,
    downstream_minor_loss_k=2.5,
    injector_total_area_m2=_injector_area_from_design(
        MORPHEUS.methane_design_flow_kg_s, LCH4, 112.0, 0.75
    ),
    injector_discharge_coefficient=0.75,
    nominal_temperature_k=112.0,
    heat_leak_w=28.0,
    color="#bd7cff",
)


@dataclass(frozen=True)
class PropulsionScenario:
    """User-controlled boundaries, bounded by the documented analysis envelope."""

    name: str = "Morpheus design point"
    lox_tank_pressure_pa: float = MORPHEUS.design_tank_pressure_pa
    methane_tank_pressure_pa: float = MORPHEUS.design_tank_pressure_pa
    lox_temperature_k: float = 90.0
    methane_temperature_k: float = 112.0
    chamber_pressure_pa: float = MORPHEUS.design_chamber_pressure_pa
    lox_valve_opening: float = MORPHEUS.design_valve_opening
    methane_valve_opening: float = MORPHEUS.design_valve_opening
    target_mixture_ratio: float = MORPHEUS.target_mixture_ratio
    mixture_ratio_tolerance: float = MORPHEUS.mixture_ratio_tolerance


@dataclass
class PropulsionSolution:
    """Combined result from the independently conserved LOX and fuel networks."""

    scenario: PropulsionScenario
    lox_network: Network
    lox: Solution
    methane_network: Network
    methane: Solution

    @property
    def lox_flow_kg_s(self) -> float:
        return self.lox.flow_into("Combustion chamber")

    @property
    def methane_flow_kg_s(self) -> float:
        return self.methane.flow_into("Combustion chamber")

    @property
    def total_flow_kg_s(self) -> float:
        return self.lox_flow_kg_s + self.methane_flow_kg_s

    @property
    def mixture_ratio(self) -> float:
        return self.lox_flow_kg_s / self.methane_flow_kg_s

    @property
    def mixture_error(self) -> float:
        return self.mixture_ratio - self.scenario.target_mixture_ratio

    @property
    def mixture_passes(self) -> bool:
        return abs(self.mixture_error) <= self.scenario.mixture_ratio_tolerance

    @property
    def estimated_thrust_n(self) -> float:
        return self.total_flow_kg_s * MORPHEUS.specific_impulse_s * GRAVITY_M_S2

    def injector_drop_ratio(self, solution: Solution) -> float:
        pressure_drop = (
            solution.node_pressures_pa["Injector inlet"]
            - self.scenario.chamber_pressure_pa
        )
        return pressure_drop / self.scenario.chamber_pressure_pa

    @property
    def lox_injector_drop_ratio(self) -> float:
        return self.injector_drop_ratio(self.lox)

    @property
    def methane_injector_drop_ratio(self) -> float:
        return self.injector_drop_ratio(self.methane)

    def injector_ratio_passes(self, ratio: float) -> bool:
        low, high = MORPHEUS.injector_band(self.scenario.chamber_pressure_pa)
        return low <= ratio <= high

    @property
    def lox_injector_passes(self) -> bool:
        return self.injector_ratio_passes(self.lox_injector_drop_ratio)

    @property
    def methane_injector_passes(self) -> bool:
        return self.injector_ratio_passes(self.methane_injector_drop_ratio)

    def physical_issues(self) -> list[str]:
        issues: list[str] = []
        for label, solution in (("LOX", self.lox), ("LCH4", self.methane)):
            path = [
                "Propellant tank",
                "Feed manifold",
                "Valve outlet",
                "Injector inlet",
                "Combustion chamber",
            ]
            pressures = [solution.node_pressures_pa[name] for name in path]
            if any(a <= b for a, b in zip(pressures, pressures[1:])):
                issues.append(f"{label} pressure does not decrease tank-to-chamber")
            if solution.flow_into("Combustion chamber") <= 0.0:
                issues.append(f"{label} has no positive chamber flow")
            if min(solution.saturation_margin_pa(name) for name in path[:-1]) <= 0.0:
                issues.append(f"{label} reaches saturation in the liquid feed path")
        return issues

    def requirement_issues(self) -> list[str]:
        issues: list[str] = []
        low, high = MORPHEUS.injector_band(self.scenario.chamber_pressure_pa)
        if not self.mixture_passes:
            issues.append("O/F is outside the 3.40 ± 0.25 analysis target")
        if not self.lox_injector_passes:
            issues.append(f"LOX injector ΔP/Pc is outside {100*low:.0f}-{100*high:.0f}%")
        if not self.methane_injector_passes:
            issues.append(f"LCH4 injector ΔP/Pc is outside {100*low:.0f}-{100*high:.0f}%")
        if abs(self.scenario.lox_valve_opening - self.scenario.methane_valve_opening) > MORPHEUS.common_valve_mismatch_limit:
            issues.append("valve commands violate the common-actuator constraint")
        if min(self.scenario.lox_tank_pressure_pa, self.scenario.methane_tank_pressure_pa) < MORPHEUS.design_tank_pressure_pa:
            issues.append("tank pressure is below the 350 psia reference minimum")
        if not MORPHEUS.lox_operating_temperature_k[0] <= self.scenario.lox_temperature_k <= MORPHEUS.lox_operating_temperature_k[1]:
            issues.append("LOX temperature is outside the single-phase operating band")
        if not MORPHEUS.methane_operating_temperature_k[0] <= self.scenario.methane_temperature_k <= MORPHEUS.methane_operating_temperature_k[1]:
            issues.append("methane temperature is outside the single-phase operating band")
        if self.scenario.chamber_pressure_pa >= 0.90 * MORPHEUS.design_chamber_pressure_pa:
            for label, solution in (("LOX", self.lox), ("LCH4", self.methane)):
                if solution.node_pressures_pa["Injector inlet"] < MORPHEUS.minimum_engine_inlet_pressure_pa:
                    issues.append(f"{label} injector inlet is below 325 psia at high power")
        return issues

    @property
    def operating_state(self) -> str:
        if not self.lox.converged or not self.methane.converged or self.physical_issues():
            return "invalid"
        if self.requirement_issues():
            return "caution"
        return "pass"

    @property
    def system_passes(self) -> bool:
        return self.operating_state == "pass"


def _calibrated_valve_loss_k(hardware: FeedHardware) -> float:
    """Fit one equivalent valve resistance to the published design point.

    Exact Morpheus pipe lengths and valve Cv data are not public. The two pipe
    losses are calculated; the remaining 25 psid from the 350 psia tank to the
    325 psia injector inlet is assigned to an equivalent throttle-valve K.
    """
    upstream = Pipe(
        hardware.upstream_length_m,
        hardware.line_diameter_m,
        minor_loss_k=hardware.upstream_minor_loss_k,
    )
    downstream = Pipe(
        hardware.downstream_length_m,
        hardware.line_diameter_m,
        minor_loss_k=hardware.downstream_minor_loss_k,
    )
    flow = hardware.design_mass_flow_kg_s
    temperature = hardware.nominal_temperature_k
    pipe_loss = upstream.pressure_loss_pa(flow, hardware.fluid, temperature)
    pipe_loss += downstream.pressure_loss_pa(flow, hardware.fluid, temperature)
    target_feed_loss = (
        MORPHEUS.design_tank_pressure_pa - MORPHEUS.minimum_engine_inlet_pressure_pa
    )
    _, _, dynamic_pressure = flow_quantities(
        flow, hardware.line_diameter_m, hardware.fluid, temperature
    )
    effective_k = max(0.05, (target_feed_loss - pipe_loss) / dynamic_pressure)
    return effective_k * MORPHEUS.design_valve_opening**2


def build_propellant_feed_network(
    hardware: FeedHardware,
    tank_pressure_pa: float,
    tank_temperature_k: float,
    chamber_pressure_pa: float,
    valve_opening: float,
) -> Network:
    """Build one pressure-fed tank-to-chamber circuit."""
    network = Network(
        f"{hardware.label} engine feed",
        hardware.fluid,
        tank_temperature_k,
    )
    network.add_node(
        Node(
            "Propellant tank",
            fixed_pressure_pa=tank_pressure_pa,
            temperature_k=tank_temperature_k,
        )
    )
    pressure_span = max(20_000.0, tank_pressure_pa - chamber_pressure_pa)
    network.add_node(Node("Feed manifold", initial_pressure_pa=tank_pressure_pa - 0.10 * pressure_span))
    network.add_node(Node("Valve outlet", initial_pressure_pa=tank_pressure_pa - 0.65 * pressure_span))
    network.add_node(Node("Injector inlet", initial_pressure_pa=tank_pressure_pa - 0.75 * pressure_span))
    network.add_node(
        Node("Combustion chamber", fixed_pressure_pa=chamber_pressure_pa)
    )

    network.add_edge(
        Edge(
            "Tank feed line",
            "Propellant tank",
            "Feed manifold",
            Pipe(
                hardware.upstream_length_m,
                hardware.line_diameter_m,
                minor_loss_k=hardware.upstream_minor_loss_k,
            ),
            heat_leak_w=0.40 * hardware.heat_leak_w,
        )
    )
    network.add_edge(
        Edge(
            "Main throttle valve",
            "Feed manifold",
            "Valve outlet",
            Valve(
                hardware.line_diameter_m,
                _calibrated_valve_loss_k(hardware),
                valve_opening,
            ),
        )
    )
    network.add_edge(
        Edge(
            "Engine feed line",
            "Valve outlet",
            "Injector inlet",
            Pipe(
                hardware.downstream_length_m,
                hardware.line_diameter_m,
                minor_loss_k=hardware.downstream_minor_loss_k,
            ),
            heat_leak_w=0.60 * hardware.heat_leak_w,
        )
    )
    network.add_edge(
        Edge(
            "Injector orifice",
            "Injector inlet",
            "Combustion chamber",
            InjectorOrifice(
                hardware.injector_total_area_m2,
                hardware.injector_discharge_coefficient,
            ),
            heat_leak_w=5.0,
        )
    )
    return network


def solve_propulsion_system(scenario: PropulsionScenario) -> PropulsionSolution:
    """Solve two separate feed networks that share only chamber pressure.

    LOX and methane never share a pipe, node, flow equation, or fluid property
    model. Their delivered mass flows are combined only after both injector
    outlets reach the common combustion-chamber boundary.
    """
    lox_network = build_propellant_feed_network(
        LOX_FEED,
        scenario.lox_tank_pressure_pa,
        scenario.lox_temperature_k,
        scenario.chamber_pressure_pa,
        scenario.lox_valve_opening,
    )
    methane_network = build_propellant_feed_network(
        METHANE_FEED,
        scenario.methane_tank_pressure_pa,
        scenario.methane_temperature_k,
        scenario.chamber_pressure_pa,
        scenario.methane_valve_opening,
    )
    lox_solution = lox_network.solve(initial_mass_flow_kg_s=MORPHEUS.lox_design_flow_kg_s)
    methane_solution = methane_network.solve(initial_mass_flow_kg_s=MORPHEUS.methane_design_flow_kg_s)
    if not lox_solution.converged or not methane_solution.converged:
        raise RuntimeError(
            "Dual feed solve did not converge: "
            f"LOX={lox_solution.residual_norm:.2e}, "
            f"LCH4={methane_solution.residual_norm:.2e}"
        )
    return PropulsionSolution(
        scenario,
        lox_network,
        lox_solution,
        methane_network,
        methane_solution,
    )


# Reporting and plots

def print_nominal_report(network: Network, solution: Solution) -> None:
    pump = network.pump_check(solution)
    print("\n" + "=" * 78)
    print("CRYONET — NOMINAL LOX TRANSFER SOLUTION")
    print("=" * 78)
    print(f"Converged: {solution.converged}")
    print(f"Scaled residual norm: {solution.residual_norm:.3e}")
    print(f"Receiver mass flow: {solution.flow_into('Vehicle tank'):.3f} kg/s")
    print(
        f"Pump NPSH: available={pump['available_m']:.2f} m, "
        f"required={pump['required_m']:.2f} m, margin={pump['margin_m']:.2f} m "
        f"({'PASS' if pump['passes'] else 'FAIL'})"
    )
    print("\nNODE STATE")
    print(f"{'Node':<18}{'P abs (kPa)':>14}{'T (K)':>12}{'P-Psat (kPa)':>17}")
    for node, pressure in solution.node_pressures_pa.items():
        temperature = solution.node_temperatures_k[node]
        margin = solution.saturation_margin_pa(node)
        print(
            f"{node:<18}{pressure / 1000:>14.1f}{temperature:>12.3f}"
            f"{margin / 1000:>17.1f}"
        )
    print("\nBRANCH RESULTS")
    print(f"{'Branch':<23}{'mdot (kg/s)':>13}{'Loss (kPa)':>13}{'Gain (kPa)':>13}")
    for result in solution.edge_results.values():
        print(
            f"{result.name:<23}{result.mass_flow_kg_s:>13.3f}"
            f"{result.pressure_loss_pa / 1000:>13.1f}"
            f"{result.pressure_gain_pa / 1000:>13.1f}"
        )


def write_solution_csv(solution: Solution, output_dir: Path) -> None:
    with (output_dir / "nominal_nodes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["node", "pressure_pa_abs", "temperature_k", "saturation_margin_pa"]
        )
        for node, pressure in solution.node_pressures_pa.items():
            writer.writerow(
                [
                    node,
                    pressure,
                    solution.node_temperatures_k[node],
                    solution.saturation_margin_pa(node),
                ]
            )
    with (output_dir / "nominal_edges.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["edge", "start", "end", "mass_flow_kg_s", "loss_pa", "gain_pa"]
        )
        for item in solution.edge_results.values():
            writer.writerow(
                [
                    item.name,
                    item.start,
                    item.end,
                    item.mass_flow_kg_s,
                    item.pressure_loss_pa,
                    item.pressure_gain_pa,
                ]
            )


def plot_network(solution: Solution, path: Path) -> None:
    pressures = np.array(
        [solution.node_pressures_pa[name] / 1000 for name in NODE_POSITIONS]
    )
    norm = Normalize(float(pressures.min()), float(pressures.max()))
    cmap = plt.get_cmap("viridis")
    network = build_transfer_network(Scenario("Nominal"))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for edge in network.edges:
        x1, y1 = NODE_POSITIONS[edge.start]
        x2, y2 = NODE_POSITIONS[edge.end]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#64748b"},
        )
        flow = solution.edge_results[edge.name].mass_flow_kg_s
        offset = 0.42 if abs(y1 - y2) < 0.05 else 0.12
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + offset,
            f"{edge.name}\n{flow:.2f} kg/s",
            ha="center",
            fontsize=8,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
        )

    for name, (x, y) in NODE_POSITIONS.items():
        pressure = solution.node_pressures_pa[name] / 1000
        temperature = solution.node_temperatures_k[name]
        color = cmap(norm(pressure))
        ax.text(
            x,
            y,
            f"{name}\n{pressure:.0f} kPa\n{temperature:.2f} K",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white" if norm(pressure) < 0.65 else "#0f172a",
            bbox={"boxstyle": "round,pad=0.55", "fc": color, "ec": "white", "lw": 2},
        )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.72
    )
    colorbar.set_label("Absolute pressure (kPa)")
    ax.set_title("CryoNet nominal LOX transfer solution", fontsize=15, fontweight="bold")
    ax.set_xlim(-1.1, 10.5)
    ax.set_ylim(-2.1, 2.1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scenarios(summaries: list[dict[str, float | str]], path: Path) -> None:
    names = [str(item["scenario"]) for item in summaries]
    flows = [float(item["receiver_flow_kg_s"]) for item in summaries]
    npsh_margins = [float(item["npsh_margin_m"]) for item in summaries]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = ax1.bar(x, flows, color="#2563eb")
    ax1.bar_label(bars, fmt="%.2f", padding=3)
    ax1.set_ylabel("Receiver mass flow (kg/s)")
    ax1.set_title("Transfer performance")
    ax1.set_xticks(x, names, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in npsh_margins]
    bars = ax2.bar(x, npsh_margins, color=colors)
    ax2.bar_label(bars, fmt="%.2f", padding=3)
    ax2.axhline(0, color="#0f172a", linewidth=1)
    ax2.set_ylabel("NPSH margin (m)")
    ax2.set_title("Pump cavitation screening")
    ax2.set_xticks(x, names, rotation=20, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("CryoNet scenario comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_valve_sweep(openings: list[float], flows: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(np.array(openings) * 100, flows, marker="o", lw=2.2, color="#7c3aed")
    ax.set_xlabel("Main valve commanded opening (%)")
    ax.set_ylabel("Receiver mass flow (kg/s)")
    ax.set_title("Main-valve restriction sensitivity", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)



# Checks and main run



def run_self_checks() -> None:
    """Small regression checks that fail loudly if core behavior changes."""
    assert abs(LOX.vapor_pressure(LOX.normal_boiling_temperature_k) - 101_325) < 1
    assert abs(LCH4.vapor_pressure(LCH4.normal_boiling_temperature_k) - 101_325) < 1
    assert abs(darcy_friction_factor(1000.0, 0.0) - 0.064) < 1.0e-12

    nominal_network, nominal = solve_scenario(Scenario("Nominal"))
    assert nominal.converged
    merge_in = (
        nominal.edge_results["Main transfer line"].mass_flow_kg_s
        + nominal.edge_results["Bypass line"].mass_flow_kg_s
    )
    merge_out = nominal.edge_results["Cryogenic coupler"].mass_flow_kg_s
    assert abs(merge_in - merge_out) < 1.0e-8

    restricted = solve_scenario(
        Scenario("Restricted", main_valve_opening=0.35)
    )[1]
    assert restricted.flow_into("Vehicle tank") < nominal.flow_into("Vehicle tank")

    low_margin_network, low_margin = solve_scenario(
        Scenario(
            "Low tank margin",
            source_pressure_pa=170_000.0,
            source_temperature_k=93.0,
        )
    )
    assert not low_margin_network.pump_check(low_margin)["passes"]

    propulsion = solve_propulsion_system(PropulsionScenario())
    assert propulsion.lox.converged and propulsion.methane.converged
    assert abs(propulsion.total_flow_kg_s - MORPHEUS.design_total_flow_kg_s) < 1.0e-8
    assert abs(propulsion.mixture_ratio - MORPHEUS.target_mixture_ratio) < 1.0e-8
    assert abs(propulsion.lox_injector_drop_ratio - 0.30) < 1.0e-8
    assert abs(propulsion.methane_injector_drop_ratio - 0.30) < 1.0e-8
    assert propulsion.lox_injector_passes
    assert propulsion.methane_injector_passes
    assert propulsion.system_passes
    assert propulsion.total_flow_kg_s > propulsion.lox_flow_kg_s
    for solution in (propulsion.lox, propulsion.methane):
        flows = [item.mass_flow_kg_s for item in solution.edge_results.values()]
        assert max(flows) - min(flows) < 1.0e-8

    # Check off-design cases

    low_pressure = solve_propulsion_system(
        PropulsionScenario(
            lox_tank_pressure_pa=330.0 * PSI_TO_PA,
            methane_tank_pressure_pa=330.0 * PSI_TO_PA,
        )
    )
    assert low_pressure.lox.converged and low_pressure.methane.converged
    assert low_pressure.operating_state == "caution"

    mismatched_valves = solve_propulsion_system(
        PropulsionScenario(lox_valve_opening=0.82, methane_valve_opening=0.72)
    )
    assert mismatched_valves.operating_state == "caution"
    print("CryoNet self-checks: PASS")


def export_static_results() -> None:
    run_self_checks()
    output_dir = Path(__file__).resolve().parent / "cryonet_results"
    output_dir.mkdir(exist_ok=True)

    result = solve_propulsion_system(PropulsionScenario())
    payload = propulsion_dashboard_payload({})
    (output_dir / "dual_feed_results.csv").write_text(
        propulsion_payload_csv(payload), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("CRYONET — NOMINAL LOX / METHANE ENGINE-FEED SOLUTION")
    print("=" * 78)
    print(f"LOX delivered:       {result.lox_flow_kg_s:8.3f} kg/s")
    print(f"Methane delivered:   {result.methane_flow_kg_s:8.3f} kg/s")
    print(f"Total propellant:    {result.total_flow_kg_s:8.3f} kg/s")
    print(f"Mixture ratio O/F:   {result.mixture_ratio:8.3f}")
    print(f"Estimated thrust:    {result.estimated_thrust_n / 1000:8.3f} kN")
    print(
        f"LOX injector dP/Pc:  {100.0 * result.lox_injector_drop_ratio:8.2f}%"
    )
    print(
        f"LCH4 injector dP/Pc: {100.0 * result.methane_injector_drop_ratio:8.2f}%"
    )
    print(f"Operating state:     {result.operating_state.upper()}")

    profiles = {
        "LOX": _feed_payload(result.lox_network, result.lox)["pressure_profile"],
        "LCH4": _feed_payload(
            result.methane_network, result.methane
        )["pressure_profile"],
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, color in (("LOX", LOX_FEED.color), ("LCH4", METHANE_FEED.color)):
        profile = profiles[label]
        ax.plot(
            [point["short"] for point in profile],
            [point["pressure"] for point in profile],
            marker="o",
            linewidth=2.5,
            label=label,
            color=color,
        )
    ax.set_ylabel("Absolute pressure (kPa)")
    ax.set_title("CryoNet nominal tank-to-chamber pressure profiles", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dual_pressure_profiles.png", dpi=190)
    plt.close(fig)

    openings = np.linspace(0.45, 0.95, 12)
    ratios = []
    lox_flows = []
    fuel_flows = []
    for common_opening in openings:
        sweep = solve_propulsion_system(
            PropulsionScenario(
                lox_valve_opening=float(common_opening),
                methane_valve_opening=float(common_opening),
            )
        )
        ratios.append(sweep.mixture_ratio)
        lox_flows.append(sweep.lox_flow_kg_s)
        fuel_flows.append(sweep.methane_flow_kg_s)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(openings * 100, ratios, color="#7c3aed", marker="o")
    ax1.axhline(result.scenario.target_mixture_ratio, color="#16a34a", ls="--")
    ax1.fill_between(
        openings * 100,
        result.scenario.target_mixture_ratio - result.scenario.mixture_ratio_tolerance,
        result.scenario.target_mixture_ratio + result.scenario.mixture_ratio_tolerance,
        color="#16a34a",
        alpha=0.12,
    )
    ax1.set(xlabel="Common throttle-valve opening (%)", ylabel="O/F mass ratio")
    ax1.set_title("Common-actuator mixture response", fontweight="bold")
    ax1.grid(alpha=0.25)
    ax2.plot(openings * 100, lox_flows, label="LOX", color=LOX_FEED.color)
    ax2.plot(openings * 100, fuel_flows, label="LCH4", color=METHANE_FEED.color)
    ax2.set(xlabel="Common throttle-valve opening (%)", ylabel="Mass flow (kg/s)")
    ax2.set_title("Delivered propellant flow", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mixture_ratio_sensitivity.png", dpi=190)
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")
    print("Created: dual_feed_results.csv, dual_pressure_profiles.png, ")
    print("         mixture_ratio_sensitivity.png")
    print("Important: this is a steady, single-phase preliminary model.")



# 7. INTERACTIVE LOCAL WEB DASHBOARD



DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CryoNet Interactive</title>
<style>
:root{--bg:#07101d;--panel:rgba(14,28,47,.88);--line:#31506f;--text:#eef6ff;--muted:#93a9bf;--cyan:#53e4ff;--blue:#4588ff;--green:#38e59c;--amber:#ffc857;--red:#ff5f72}
*{box-sizing:border-box}body{margin:0;color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at 15% 15%,rgba(35,111,170,.25),transparent 30%),radial-gradient(circle at 80% 5%,rgba(73,57,150,.22),transparent 28%),linear-gradient(150deg,#030812 0%,var(--bg) 52%,#091a2c 100%);min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.32;background-image:radial-gradient(circle at 20% 30%,#fff 0 1px,transparent 1.5px),radial-gradient(circle at 70% 15%,#fff 0 1px,transparent 1.5px),radial-gradient(circle at 90% 60%,#fff 0 1px,transparent 1.5px),radial-gradient(circle at 35% 80%,#fff 0 1px,transparent 1.5px);background-size:270px 240px,310px 290px,240px 320px,360px 270px}
.shell{width:min(1500px,96vw);margin:0 auto;padding:26px 0 50px;position:relative}header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:22px}.eyebrow{color:var(--cyan);letter-spacing:.19em;font-size:12px;font-weight:700;text-transform:uppercase}h1{font-size:clamp(30px,4vw,56px);line-height:.98;margin:8px 0;letter-spacing:-.04em}.subtitle{color:var(--muted);margin:0;max-width:720px}.live{display:inline-flex;align-items:center;gap:8px;color:var(--green);font-size:13px;letter-spacing:.08em}.live:before{content:"";width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 18px var(--green)}
.layout{display:grid;grid-template-columns:280px 1fr;gap:18px;align-items:start}.panel,.metric,.status{background:var(--panel);border:1px solid rgba(118,167,204,.18);border-radius:18px;box-shadow:0 18px 50px rgba(0,0,0,.22);backdrop-filter:blur(16px)}.controls{padding:20px;position:sticky;top:18px}.controls h2,.panel h2{margin:0 0 6px;font-size:16px}.hint{margin:0 0 22px;color:var(--muted);font-size:12px}.control{margin:19px 0}.control label{display:flex;justify-content:space-between;gap:10px;font-size:12px;color:var(--muted);margin-bottom:8px}.control output{color:var(--text);font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--cyan)}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:20px}button,.download{border:1px solid rgba(108,174,221,.25);border-radius:10px;padding:10px 12px;color:var(--text);background:rgba(38,72,105,.5);font:inherit;font-size:12px;text-decoration:none;text-align:center;cursor:pointer}.download{background:linear-gradient(135deg,#147fa1,#3159cb);border:none}.main{min-width:0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}.metric{padding:16px}.metric span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.metric strong{display:block;margin-top:6px;font-size:27px;font-variant-numeric:tabular-nums}.metric small{color:var(--muted)}
.status{padding:12px 16px;margin-bottom:12px;display:flex;justify-content:space-between;gap:14px;align-items:center}.status.pass{border-color:rgba(56,229,156,.4);background:rgba(14,71,62,.62)}.status.fail{border-color:rgba(255,95,114,.5);background:rgba(85,25,39,.65)}.status-title{font-weight:700}.status-detail{color:var(--muted);font-size:12px}.panel{padding:18px;margin-bottom:12px}.section-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.legend{color:var(--muted);font-size:11px}.network-wrap{overflow-x:auto}#network{display:block;width:100%;min-width:780px;height:auto}.flow-line{fill:none;stroke:var(--cyan);stroke-width:4;stroke-dasharray:10 9;animation:flow 1.1s linear infinite;filter:drop-shadow(0 0 5px rgba(83,228,255,.45))}.flow-line.bypass{stroke:var(--blue);opacity:.8}@keyframes flow{to{stroke-dashoffset:-38}}.node-box{fill:#102943;stroke:rgba(115,194,242,.35);stroke-width:1.5}.node-name{fill:var(--muted);font-size:12px;text-anchor:middle}.node-value{fill:var(--text);font-size:15px;font-weight:700;text-anchor:middle}.edge-label{fill:var(--text);font-size:11px;text-anchor:middle}.edge-sub{fill:var(--muted);font-size:10px;text-anchor:middle}
.lower{display:grid;grid-template-columns:1.45fr 1fr;gap:12px}#pressureChart{display:block;width:100%;height:auto}.chart-axis{stroke:#44627e;stroke-width:1}.chart-grid{stroke:rgba(105,144,177,.16);stroke-width:1}.chart-line{fill:none;stroke:var(--cyan);stroke-width:3;filter:drop-shadow(0 0 5px rgba(83,228,255,.35))}.chart-dot{fill:var(--bg);stroke:var(--cyan);stroke-width:3}.chart-label{fill:var(--muted);font-size:10px;text-anchor:middle}.chart-value{fill:var(--text);font-size:10px;text-anchor:middle}.split-row{margin:17px 0}.split-head{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:7px}.bar-track{height:11px;border-radius:99px;background:rgba(125,159,188,.15);overflow:hidden}.bar-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--cyan),var(--blue));transition:width .35s ease}.bar-fill.bypass{background:linear-gradient(90deg,#5968ff,#a35dff)}.gauge{margin-top:22px}.gauge-scale{height:12px;border-radius:99px;background:linear-gradient(90deg,var(--red) 0 17%,var(--amber) 17% 30%,var(--green) 30% 100%);position:relative}.pointer{position:absolute;top:-6px;width:3px;height:24px;background:white;box-shadow:0 0 8px white;transition:left .35s ease}.gauge-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-top:7px}.losses{margin-top:22px}.loss-row{display:grid;grid-template-columns:118px 1fr 55px;gap:8px;align-items:center;margin:8px 0;font-size:11px}.loss-name{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.loss-track{height:7px;background:rgba(125,159,188,.13);border-radius:99px;overflow:hidden}.loss-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:inherit}.loss-value{text-align:right;font-variant-numeric:tabular-nums}.footer{color:var(--muted);font-size:11px;margin-top:16px;line-height:1.6}
@media(max-width:980px){.layout{grid-template-columns:1fr}.controls{position:static}.metrics{grid-template-columns:repeat(2,1fr)}.lower{grid-template-columns:1fr}}@media(max-width:560px){.metrics{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}@media(prefers-reduced-motion:reduce){.flow-line{animation:none}}
</style>
</head>
<body>
<main class="shell">
<header><div><div class="eyebrow">Cryogenic fluid systems</div><h1>CryoNet // Digital Twin</h1><p class="subtitle">Interactive LOX transfer network analysis from storage tank to receiving vehicle.</p></div><div class="live">SOLVER ONLINE</div></header>
<div class="layout">
<aside class="panel controls"><h2>Operating conditions</h2><p class="hint">Move a control to re-solve the entire network.</p>
<div class="control"><label for="sourceP"><span>Storage tank pressure</span><output id="sourcePOut">450 kPa</output></label><input id="sourceP" type="range" min="150" max="800" step="5" value="450"></div>
<div class="control"><label for="temp"><span>LOX temperature</span><output id="tempOut">90.0 K</output></label><input id="temp" type="range" min="87" max="100" step="0.1" value="90"></div>
<div class="control"><label for="valve"><span>Main valve opening</span><output id="valveOut">100%</output></label><input id="valve" type="range" min="20" max="100" step="1" value="100"></div>
<div class="control"><label for="pump"><span>Pump health</span><output id="pumpOut">100%</output></label><input id="pump" type="range" min="50" max="110" step="1" value="100"></div>
<div class="control"><label for="receiverP"><span>Vehicle tank pressure</span><output id="receiverPOut">300 kPa</output></label><input id="receiverP" type="range" min="200" max="600" step="5" value="300"></div>
<div class="buttons"><button id="reset" type="button">Reset</button><a id="download" class="download" href="/download">Export CSV</a></div><p class="footer">Steady, one-dimensional, single-phase preliminary model. Pressures are absolute.</p></aside>
<section class="main">
<div class="metrics"><div class="metric"><span>Receiver flow</span><strong id="flowMetric">—</strong><small>kg/s</small></div><div class="metric"><span>Pump inlet</span><strong id="pinMetric">—</strong><small>kPa abs</small></div><div class="metric"><span>Pump outlet</span><strong id="poutMetric">—</strong><small>kPa abs</small></div><div class="metric"><span>NPSH margin</span><strong id="npshMetric">—</strong><small>m</small></div></div>
<div id="status" class="status pass" aria-live="polite"><div><div id="statusTitle" class="status-title">Solving network…</div><div id="statusDetail" class="status-detail"></div></div><div id="residual" class="status-detail"></div></div>
<section class="panel"><div class="section-head"><h2>Live transfer architecture</h2><span class="legend">Animated paths show solved flow direction</span></div><div class="network-wrap">
<svg id="network" viewBox="0 0 1120 500" role="img" aria-label="Live LOX transfer network"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#53e4ff"/></marker></defs>
<path class="flow-line" d="M140 250L190 250" marker-end="url(#arrow)"/><path class="flow-line" d="M300 250L350 250" marker-end="url(#arrow)"/><path class="flow-line" d="M460 240L520 150" marker-end="url(#arrow)"/><path class="flow-line" d="M630 150L700 240" marker-end="url(#arrow)"/><path class="flow-line bypass" d="M460 260L520 350" marker-end="url(#arrow)"/><path class="flow-line bypass" d="M630 350L700 260" marker-end="url(#arrow)"/><path class="flow-line" d="M810 250L850 250" marker-end="url(#arrow)"/><path class="flow-line" d="M960 250L1000 250" marker-end="url(#arrow)"/>
<g transform="translate(30 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">STORAGE TANK</text><text id="n-storage" class="node-value" x="55" y="51">—</text></g><g transform="translate(190 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">PUMP INLET</text><text id="n-pin" class="node-value" x="55" y="51">—</text></g><g transform="translate(350 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">PUMP OUTLET</text><text id="n-pout" class="node-value" x="55" y="51">—</text></g><g transform="translate(520 115)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">MAIN PATH</text><text id="n-main" class="node-value" x="55" y="51">—</text></g><g transform="translate(520 315)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">BYPASS</text><text id="n-bypass" class="node-value" x="55" y="51">—</text></g><g transform="translate(700 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">MERGE</text><text id="n-merge" class="node-value" x="55" y="51">—</text></g><g transform="translate(850 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">COUPLER</text><text id="n-coupler" class="node-value" x="55" y="51">—</text></g><g transform="translate(1000 215)"><rect class="node-box" width="110" height="70" rx="16"/><text class="node-name" x="55" y="27">VEHICLE</text><text id="n-vehicle" class="node-value" x="55" y="51">—</text></g>
<text class="edge-label" x="165" y="225">SUCTION</text><text id="e-suction" class="edge-sub" x="165" y="301">—</text><text class="edge-label" x="325" y="225">PUMP</text><text id="e-pump" class="edge-sub" x="325" y="301">—</text><text class="edge-label" x="490" y="175">MAIN</text><text id="e-main" class="edge-sub" x="490" y="190">—</text><text class="edge-label" x="490" y="332">BYPASS</text><text id="e-bypass" class="edge-sub" x="490" y="347">—</text><text class="edge-label" x="830" y="225">COUPLER</text><text id="e-coupler" class="edge-sub" x="830" y="301">—</text><text class="edge-label" x="980" y="225">FILL</text><text id="e-fill" class="edge-sub" x="980" y="301">—</text></svg>
</div></section>
<div class="lower"><section class="panel"><div class="section-head"><h2>Pressure along main flow path</h2><span class="legend">kPa absolute</span></div><svg id="pressureChart" viewBox="0 0 700 310" role="img" aria-label="Pressure profile chart"></svg></section><section class="panel"><h2>Flow distribution & safety</h2><div class="split-row"><div class="split-head"><span>Main path</span><span id="mainSplit">—</span></div><div class="bar-track"><div id="mainBar" class="bar-fill"></div></div></div><div class="split-row"><div class="split-head"><span>Bypass path</span><span id="bypassSplit">—</span></div><div class="bar-track"><div id="bypassBar" class="bar-fill bypass"></div></div></div><div class="gauge"><div class="split-head"><span>NPSH safety margin</span><span id="gaugeValue">—</span></div><div class="gauge-scale"><div id="pointer" class="pointer"></div></div><div class="gauge-labels"><span>−5 m</span><span>0</span><span>30 m</span></div></div><div class="losses"><div class="split-head"><span>Largest component losses</span><span>kPa</span></div><div id="lossBars"></div></div></section></div>
<p class="footer">CryoNet solves node pressures and branch mass flows simultaneously using mass conservation and component pressure relations. This dashboard is a preliminary digital twin—not a replacement for verified cryogenic test data or two-phase analysis.</p></section></div></main>
<script>
const controls={sourceP:document.getElementById('sourceP'),temp:document.getElementById('temp'),valve:document.getElementById('valve'),pump:document.getElementById('pump'),receiverP:document.getElementById('receiverP')};
const sourcePOut=document.getElementById('sourcePOut'),tempOut=document.getElementById('tempOut'),valveOut=document.getElementById('valveOut'),pumpOut=document.getElementById('pumpOut'),receiverPOut=document.getElementById('receiverPOut');
const flowMetric=document.getElementById('flowMetric'),pinMetric=document.getElementById('pinMetric'),poutMetric=document.getElementById('poutMetric'),npshMetric=document.getElementById('npshMetric');
const statusBox=document.getElementById('status'),statusTitleEl=document.getElementById('statusTitle'),statusDetailEl=document.getElementById('statusDetail'),residualEl=document.getElementById('residual');
const mainSplit=document.getElementById('mainSplit'),bypassSplit=document.getElementById('bypassSplit'),mainBar=document.getElementById('mainBar'),bypassBar=document.getElementById('bypassBar'),gaugeValue=document.getElementById('gaugeValue'),pointer=document.getElementById('pointer'),pressureChart=document.getElementById('pressureChart'),lossBars=document.getElementById('lossBars');
const defaults={sourceP:450,temp:90,valve:100,pump:100,receiverP:300};let timer;
function params(){return new URLSearchParams({source_pressure:controls.sourceP.value,temperature:controls.temp.value,valve:controls.valve.value,pump:controls.pump.value,receiver_pressure:controls.receiverP.value})}
function updateLabels(){sourcePOut.value=`${controls.sourceP.value} kPa`;tempOut.value=`${Number(controls.temp.value).toFixed(1)} K`;valveOut.value=`${controls.valve.value}%`;pumpOut.value=`${controls.pump.value}%`;receiverPOut.value=`${controls.receiverP.value} kPa`;document.getElementById('download').href=`/download?${params()}`}
Object.values(controls).forEach(el=>el.addEventListener('input',()=>{updateLabels();clearTimeout(timer);timer=setTimeout(solve,180)}));document.getElementById('reset').addEventListener('click',()=>{Object.entries(defaults).forEach(([k,v])=>controls[k].value=v);updateLabels();solve()});
async function solve(){statusTitleEl.textContent='Solving network…';try{const response=await fetch(`/api?${params()}`);const data=await response.json();if(!response.ok)throw new Error(data.error||'Solver error');render(data)}catch(error){statusBox.className='status fail';statusTitleEl.textContent='Solver could not evaluate this point';statusDetailEl.textContent=error.message}}
function render(data){const s=data.summary;flowMetric.textContent=s.receiver_flow.toFixed(3);pinMetric.textContent=s.pump_inlet.toFixed(1);poutMetric.textContent=s.pump_outlet.toFixed(1);npshMetric.textContent=s.npsh_margin.toFixed(2);statusBox.className=`status ${s.passes?'pass':'fail'}`;statusTitleEl.textContent=s.passes?'Single-phase operating point passes NPSH screen':'Operating point fails NPSH screen';statusDetailEl.textContent=s.passes?`Available ${s.npsh_available.toFixed(2)} m vs required ${s.npsh_required.toFixed(2)} m`:'Reject this point or evaluate with higher-fidelity cryogenic and pump data.';residualEl.textContent=`Residual ${s.residual.toExponential(2)}`;
const nodeMap=Object.fromEntries(data.nodes.map(n=>[n.name,n]));const ids={'Storage tank':'n-storage','Pump inlet':'n-pin','Pump outlet':'n-pout','Main branch':'n-main','Bypass branch':'n-bypass','Branch merge':'n-merge','Coupler outlet':'n-coupler','Vehicle tank':'n-vehicle'};Object.entries(ids).forEach(([name,id])=>document.getElementById(id).textContent=`${nodeMap[name].pressure.toFixed(0)} kPa`);const edgeMap=Object.fromEntries(data.edges.map(e=>[e.name,e]));document.getElementById('e-suction').textContent=`${edgeMap['Suction line'].flow.toFixed(2)} kg/s`;document.getElementById('e-pump').textContent=`+${edgeMap['Transfer pump'].gain.toFixed(0)} kPa`;document.getElementById('e-main').textContent=`${edgeMap['Main transfer line'].flow.toFixed(2)} kg/s`;document.getElementById('e-bypass').textContent=`${edgeMap['Bypass line'].flow.toFixed(2)} kg/s`;document.getElementById('e-coupler').textContent=`−${edgeMap['Cryogenic coupler'].loss.toFixed(0)} kPa`;document.getElementById('e-fill').textContent=`−${edgeMap['Vehicle fill line'].loss.toFixed(0)} kPa`;
const main=edgeMap['Main transfer line'].flow,bypass=edgeMap['Bypass line'].flow,total=main+bypass,mainPct=100*main/total,bypassPct=100*bypass/total;mainSplit.textContent=`${main.toFixed(2)} kg/s · ${mainPct.toFixed(0)}%`;bypassSplit.textContent=`${bypass.toFixed(2)} kg/s · ${bypassPct.toFixed(0)}%`;mainBar.style.width=`${mainPct}%`;bypassBar.style.width=`${bypassPct}%`;gaugeValue.textContent=`${s.npsh_margin.toFixed(2)} m`;pointer.style.left=`${Math.max(0,Math.min(100,(s.npsh_margin+5)/35*100))}%`;drawPressure(data.pressure_profile);drawLosses(data.edges.filter(e=>e.loss>0).sort((a,b)=>b.loss-a.loss).slice(0,5))}
function drawPressure(points){const svg=pressureChart;svg.innerHTML='';const NS='http://www.w3.org/2000/svg',W=700,H=310,L=55,R=20,T=24,B=62,values=points.map(p=>p.pressure),min=Math.min(...values),max=Math.max(...values),pad=Math.max(20,(max-min)*.12),yMin=min-pad,yMax=max+pad,x=i=>L+i*(W-L-R)/(points.length-1),y=v=>T+(yMax-v)*(H-T-B)/(yMax-yMin);for(let i=0;i<4;i++){const gy=T+i*(H-T-B)/3,line=document.createElementNS(NS,'line');line.setAttribute('x1',L);line.setAttribute('x2',W-R);line.setAttribute('y1',gy);line.setAttribute('y2',gy);line.setAttribute('class','chart-grid');svg.appendChild(line);const tx=document.createElementNS(NS,'text');tx.setAttribute('x',L-8);tx.setAttribute('y',gy+4);tx.setAttribute('text-anchor','end');tx.setAttribute('class','chart-label');tx.textContent=(yMax-i*(yMax-yMin)/3).toFixed(0);svg.appendChild(tx)}const axis=document.createElementNS(NS,'line');axis.setAttribute('x1',L);axis.setAttribute('x2',W-R);axis.setAttribute('y1',H-B);axis.setAttribute('y2',H-B);axis.setAttribute('class','chart-axis');svg.appendChild(axis);const poly=document.createElementNS(NS,'polyline');poly.setAttribute('points',points.map((p,i)=>`${x(i)},${y(p.pressure)}`).join(' '));poly.setAttribute('class','chart-line');svg.appendChild(poly);points.forEach((p,i)=>{const dot=document.createElementNS(NS,'circle');dot.setAttribute('cx',x(i));dot.setAttribute('cy',y(p.pressure));dot.setAttribute('r',5);dot.setAttribute('class','chart-dot');svg.appendChild(dot);const val=document.createElementNS(NS,'text');val.setAttribute('x',x(i));val.setAttribute('y',y(p.pressure)-12);val.setAttribute('class','chart-value');val.textContent=p.pressure.toFixed(0);svg.appendChild(val);const lab=document.createElementNS(NS,'text');lab.setAttribute('x',x(i));lab.setAttribute('y',H-B+22);lab.setAttribute('class','chart-label');lab.textContent=p.short;svg.appendChild(lab)})}
function drawLosses(edges){const max=Math.max(...edges.map(e=>e.loss));lossBars.innerHTML=edges.map(e=>`<div class="loss-row"><span class="loss-name">${e.name}</span><div class="loss-track"><div class="loss-fill" style="width:${100*e.loss/max}%"></div></div><span class="loss-value">${e.loss.toFixed(1)}</span></div>`).join('')}
updateLabels();solve();
</script></body></html>'''


# A self-contained dashboard is embedded so the entire portfolio project still
# ships as one Python file with no front-end build step.
DUAL_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CryoNet — Dual-Propellant Feed Solver</title>
<style>
:root{--bg:#050914;--panel:rgba(14,24,42,.9);--line:#29405d;--text:#f3f8ff;--muted:#91a5bd;--lox:#48ddff;--fuel:#bd7cff;--green:#35e49a;--amber:#ffc857;--red:#ff6478}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:radial-gradient(circle at 10% 10%,#0c3853 0,transparent 29%),radial-gradient(circle at 88% 8%,#34205e 0,transparent 28%),linear-gradient(145deg,#03060d,var(--bg) 55%,#071523)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:radial-gradient(#fff 0 1px,transparent 1.4px);background-size:170px 150px}
.shell{width:min(1560px,96vw);margin:auto;padding:24px 0 45px;position:relative}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:18px}.eyebrow{color:var(--lox);font-size:11px;font-weight:800;letter-spacing:.2em;text-transform:uppercase}.top h1{font-size:clamp(31px,4vw,57px);letter-spacing:-.045em;line-height:1;margin:7px 0}.top p{color:var(--muted);margin:0}.live{color:var(--green);font-size:12px;letter-spacing:.1em}.live:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:var(--green);box-shadow:0 0 16px var(--green)}
.layout{display:grid;grid-template-columns:292px 1fr;gap:15px}.panel,.metric,.status{background:var(--panel);border:1px solid rgba(130,170,210,.17);border-radius:18px;box-shadow:0 17px 48px rgba(0,0,0,.22);backdrop-filter:blur(14px)}.controls{padding:18px;position:sticky;top:15px;max-height:calc(100vh - 30px);overflow:auto}.controls h2,.panel h2{font-size:15px;margin:0}.hint,.note{color:var(--muted);font-size:11px;line-height:1.55}.group{margin-top:19px;padding-top:15px;border-top:1px solid rgba(130,170,210,.13)}.group-title{display:flex;gap:8px;align-items:center;font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.dot{width:8px;height:8px;border-radius:50%;background:var(--lox);box-shadow:0 0 10px var(--lox)}.dot.fuel{background:var(--fuel);box-shadow:0 0 10px var(--fuel)}.control{margin:14px 0}.control label{display:flex;justify-content:space-between;gap:9px;color:var(--muted);font-size:11px;margin-bottom:6px}.control output{color:var(--text);font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--lox)}.fuel-control input{accent-color:var(--fuel)}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:17px}button,.download{font:inherit;font-size:11px;color:var(--text);padding:10px;border-radius:10px;border:1px solid rgba(130,170,210,.23);background:#152941;text-align:center;text-decoration:none;cursor:pointer}.download{background:linear-gradient(135deg,#167f9b,#6a42bc);border:0}
.main{min-width:0}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:10px}.metric{padding:14px}.metric span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric strong{display:block;font-size:25px;margin-top:5px;font-variant-numeric:tabular-nums}.metric small{color:var(--muted);font-size:10px}.lox-text{color:var(--lox)}.fuel-text{color:var(--fuel)}
.status{padding:11px 15px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;gap:15px}.status.pass{background:rgba(10,72,59,.62);border-color:rgba(53,228,154,.38)}.status.caution{background:rgba(83,61,17,.67);border-color:rgba(255,200,87,.46)}.status.invalid{background:rgba(87,26,40,.65);border-color:rgba(255,100,120,.45)}.status strong{font-size:13px}.status span{color:var(--muted);font-size:11px}.panel{padding:16px;margin-bottom:10px}.head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px}.legend{display:flex;gap:13px;color:var(--muted);font-size:10px}.key:before{content:"";display:inline-block;width:16px;height:3px;border-radius:3px;margin-right:5px;vertical-align:middle;background:var(--lox)}.key.fuel:before{background:var(--fuel)}
.network-wrap{overflow:auto}#network{width:100%;height:auto;min-width:840px;display:block}.pipe{fill:none;stroke-width:5;stroke-dasharray:11 9;animation:flow 1.05s linear infinite;filter:drop-shadow(0 0 5px currentColor)}.pipe.lox{stroke:var(--lox);color:var(--lox)}.pipe.fuel{stroke:var(--fuel);color:var(--fuel)}@keyframes flow{to{stroke-dashoffset:-40}}.node{fill:#10243a;stroke:#385b7b;stroke-width:1.4}.node.fuel{stroke:#7954a5}.node-name{fill:var(--muted);font-size:10px;text-anchor:middle}.node-value{fill:var(--text);font-size:14px;font-weight:800;text-anchor:middle}.lane{font-size:13px;font-weight:900;letter-spacing:.1em}.edge-text{fill:var(--muted);font-size:10px;text-anchor:middle}.engine{fill:url(#engineGlow);stroke:#ffbd66;stroke-width:2}.flame{fill:#ff8a45;filter:drop-shadow(0 0 12px #ff9a42);animation:pulse .8s ease-in-out infinite alternate}@keyframes pulse{to{opacity:.58;transform:translateY(4px)}}
.lower{display:grid;grid-template-columns:1.35fr 1fr;gap:10px}#chart{width:100%;height:auto;display:block}.grid{stroke:rgba(132,163,193,.15)}.axis{stroke:#48647e}.chart-label{fill:var(--muted);font-size:9px;text-anchor:middle}.chart-value{fill:var(--text);font-size:9px;text-anchor:middle}.chart-lox,.chart-fuel{fill:none;stroke-width:3}.chart-lox{stroke:var(--lox)}.chart-fuel{stroke:var(--fuel);stroke-dasharray:9 6}.dot-lox{fill:var(--bg);stroke:var(--lox);stroke-width:2.5}.dot-fuel{fill:var(--bg);stroke:var(--fuel);stroke-width:2.5}
.balance{margin:16px 0}.balance-head{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-bottom:7px}.mixbar{height:16px;display:flex;border-radius:99px;overflow:hidden;background:#16273b}.mix-lox{background:linear-gradient(90deg,#1583af,var(--lox))}.mix-fuel{background:linear-gradient(90deg,var(--fuel),#784ab3)}.safety{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}.safety-card{padding:12px;background:rgba(7,16,29,.55);border-radius:12px}.safety-card span{display:block;color:var(--muted);font-size:10px}.safety-card strong{font-size:19px}.pass-text{color:var(--green)}.fail-text{color:var(--red)}.losses{margin-top:16px}.loss-row{display:grid;grid-template-columns:84px 1fr 47px;gap:7px;align-items:center;margin:7px 0;font-size:10px}.loss-name{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{height:6px;background:#172a3e;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--lox),var(--fuel));border-radius:99px}.loss-value{text-align:right}.explain{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.explain article{padding:13px;background:rgba(8,18,32,.55);border-radius:12px}.explain b{font-size:11px}.explain p{color:var(--muted);font-size:10px;line-height:1.5;margin:6px 0 0}.footer{color:var(--muted);font-size:10px;line-height:1.6;margin-top:12px}.footer a{color:var(--lox)}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.controls{position:static;max-height:none}.metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:720px){.top{align-items:start;flex-direction:column}.metrics{grid-template-columns:repeat(2,1fr)}.lower,.explain{grid-template-columns:1fr}}@media(max-width:450px){.metrics{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){.pipe,.flame{animation:none}}
</style></head><body><main class="shell">
<header class="top"><div><div class="eyebrow">NASA Morpheus requirements case</div><h1>CryoNet // Engine Feed Solver</h1><p>A constrained pressure-fed LOX/methane network. Change real operating inputs and see whether the result stays inside the documented envelope.</p></div><div class="live">TWO SOLVERS ONLINE</div></header>
<div class="layout"><aside class="panel controls"><h2>Operating conditions</h2><p class="hint">The solver remains interactive, but every result is graded against the Morpheus design basis.</p>
<div class="group"><div class="group-title"><i class="dot"></i>LOX circuit</div>
<div class="control"><label>Tank pressure <output id="loxPOut"></output></label><input id="loxP" type="range" min="2100" max="2700" step="1" value="2414"></div>
<div class="control"><label>Temperature <output id="loxTOut"></output></label><input id="loxT" type="range" min="84" max="105" step=".1" value="90"></div>
</div>
<div class="group fuel-control"><div class="group-title"><i class="dot fuel"></i>Methane circuit</div>
<div class="control"><label>Tank pressure <output id="fuelPOut"></output></label><input id="fuelP" type="range" min="2100" max="2700" step="1" value="2414"></div>
<div class="control"><label>Temperature <output id="fuelTOut"></output></label><input id="fuelT" type="range" min="100" max="128" step=".1" value="112"></div>
</div>
<div class="group"><div class="group-title">Shared engine boundary</div>
<div class="control"><label>Common throttle valve <output id="valveOut"></output></label><input id="valve" type="range" min="35" max="95" step="1" value="82"></div>
<div class="control"><label>Chamber pressure <output id="chamberOut"></output></label><input id="chamber" type="range" min="550" max="1800" step="5" value="1724"></div></div>
<div class="buttons"><button id="reset">Design point</button><a id="download" class="download" href="/download">Export CSV</a></div><p class="note">Absolute pressures. Fixed reference: 4,200 lbf, 215 s, O/F 3.40, 4:1 throttle. Green means requirement-compliant; amber is solvable off-design; red is physically invalid.</p></aside>
<section class="main"><div class="metrics">
<div class="metric"><span>LOX flow</span><strong id="loxFlow" class="lox-text">—</strong><small>kg/s</small></div><div class="metric"><span>Methane flow</span><strong id="fuelFlow" class="fuel-text">—</strong><small>kg/s</small></div><div class="metric"><span>Total flow</span><strong id="totalFlow">—</strong><small>kg/s</small></div><div class="metric"><span>Mixture ratio</span><strong id="ofRatio">—</strong><small>O/F by mass</small></div><div class="metric"><span>Estimated thrust</span><strong id="thrustMetric">—</strong><small>kN at 215 s</small></div></div>
<div id="status" class="status pass"><div><strong id="statusTitle">Solving both circuits…</strong><br><span id="statusDetail"></span></div><span id="residual"></span></div>
<section class="panel"><div class="head"><h2>Live dual-propellant architecture</h2><div class="legend"><span class="key">LOX</span><span class="key fuel">LCH4</span></div></div><div class="network-wrap">
<svg id="network" viewBox="0 0 1120 440" aria-label="LOX and methane feed networks converging on an engine"><defs><linearGradient id="engineGlow" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#50341f"/><stop offset="1" stop-color="#1a2431"/></linearGradient><marker id="aL" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#48ddff"/></marker><marker id="aF" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#bd7cff"/></marker></defs>
<text x="20" y="73" class="lane" fill="#48ddff">LOX PATH</text><text x="20" y="313" class="lane" fill="#bd7cff">LCH4 PATH</text>
<path class="pipe lox" d="M165 100H250M365 100H450M565 100H650M765 100C850 100 850 188 925 188" marker-end="url(#aL)"/><path class="pipe fuel" d="M165 340H250M365 340H450M565 340H650M765 340C850 340 850 252 925 252" marker-end="url(#aF)"/>
<g transform="translate(45 68)"><rect class="node" width="120" height="64" rx="15"/><text class="node-name" x="60" y="25">LOX TANK</text><text id="loxTank" class="node-value" x="60" y="48">—</text></g><g transform="translate(250 68)"><rect class="node" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">FEED MANIFOLD</text><text id="loxFeed" class="node-value" x="57" y="48">—</text></g><g transform="translate(450 68)"><rect class="node" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">VALVE OUTLET</text><text id="loxValve" class="node-value" x="57" y="48">—</text></g><g transform="translate(650 68)"><rect class="node" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">INJECTOR INLET</text><text id="loxInjector" class="node-value" x="57" y="48">—</text></g>
<g transform="translate(45 308)"><rect class="node fuel" width="120" height="64" rx="15"/><text class="node-name" x="60" y="25">LCH4 TANK</text><text id="fuelTank" class="node-value" x="60" y="48">—</text></g><g transform="translate(250 308)"><rect class="node fuel" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">FEED MANIFOLD</text><text id="fuelFeed" class="node-value" x="57" y="48">—</text></g><g transform="translate(450 308)"><rect class="node fuel" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">VALVE OUTLET</text><text id="fuelValve" class="node-value" x="57" y="48">—</text></g><g transform="translate(650 308)"><rect class="node fuel" width="115" height="64" rx="15"/><text class="node-name" x="57" y="25">INJECTOR INLET</text><text id="fuelInjector" class="node-value" x="57" y="48">—</text></g>
<g transform="translate(925 155)"><path class="engine" d="M0 0h130l-20 130H20z"/><text class="node-name" x="65" y="38">COMBUSTION</text><text class="node-name" x="65" y="53">CHAMBER</text><text id="engineValue" class="node-value" x="65" y="82">—</text><path class="flame" d="M42 130h46l-12 75-11-17-11 17z"/></g>
<text id="loxPipe" class="edge-text" x="408" y="86">—</text><text id="fuelPipe" class="edge-text" x="408" y="326">—</text></svg></div></section>
<div class="lower"><section class="panel"><div class="head"><h2>Two separate tank-to-chamber pressure profiles</h2><div class="legend"><span class="key">LOX</span><span class="key fuel">LCH4</span><span>kPa absolute</span></div></div><svg id="chart" viewBox="0 0 720 320" aria-label="Separate LOX and methane pressure profiles"></svg></section>
<section class="panel"><h2>Propellant balance & requirement screens</h2><div class="balance"><div class="balance-head"><span>Delivered mass split</span><span id="splitText">—</span></div><div class="mixbar"><div id="mixLox" class="mix-lox"></div><div id="mixFuel" class="mix-fuel"></div></div></div><div class="safety"><div class="safety-card"><span>LOX injector ΔP/Pc</span><strong id="loxInjectorRatio">—</strong></div><div class="safety-card"><span>LCH4 injector ΔP/Pc</span><strong id="fuelInjectorRatio">—</strong></div><div class="safety-card"><span>LOX liquid margin</span><strong id="loxMargin">—</strong></div><div class="safety-card"><span>LCH4 liquid margin</span><strong id="fuelMargin">—</strong></div><div class="safety-card"><span>LOX feed ΔP</span><strong id="loxFeedDrop">—</strong></div><div class="safety-card"><span>LCH4 feed ΔP</span><strong id="fuelFeedDrop">—</strong></div></div><div class="losses"><div class="balance-head"><span>Largest pressure losses</span><span>kPa</span></div><div id="lossBars"></div></div></section></div>
<section class="panel"><div class="explain"><article><b>1 · General solver</b><p>Each node pressure and branch flow is solved from mass conservation plus pipe, valve, and injector equations.</p></article><article><b>2 · Real reference case</b><p>The nominal point targets 4,200 lbf, 215 s, 350 psia tanks, 325 psia injector inlets, and a 250 psia chamber.</p></article><article><b>3 · Constrained exploration</b><p>Green is inside the design envelope, amber is a solvable off-design case, and red means a physical constraint failed.</p></article><article><b>4 · Honest calibration</b><p>Unpublished line and valve details are represented by equivalent resistance fitted to the reference pressure budget.</p></article></div></section><p class="footer">CryoNet follows NASA GFSSP's node/branch conservation approach but is independently written in Python. Sources: <a href="https://ntrs.nasa.gov/citations/20110014012">Morpheus requirements</a>, <a href="https://ntrs.nasa.gov/citations/20140009917">Morpheus injector and throttling data</a>, and <a href="https://ntrs.nasa.gov/citations/19760023196">NASA SP-8089 injector criteria</a>. This remains a steady, one-dimensional, single-phase educational reconstruction—not flight software or an exact hardware replica.</p></section></div></main>
<script>
const ids=['loxP','loxT','fuelP','fuelT','valve','chamber'];const C=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));const D={loxP:2414,loxT:90,fuelP:2414,fuelT:112,valve:82,chamber:1724};let timer;
function params(){return new URLSearchParams({lox_pressure:C.loxP.value,lox_temperature:C.loxT.value,fuel_pressure:C.fuelP.value,fuel_temperature:C.fuelT.value,valve:C.valve.value,chamber_pressure:C.chamber.value})}
function labels(){for(const [id,suffix,digits] of [['loxP',' kPa',0],['loxT',' K',1],['fuelP',' kPa',0],['fuelT',' K',1],['valve','%',0],['chamber',' kPa',0]])document.getElementById(id+'Out').value=Number(C[id].value).toFixed(digits)+suffix;document.getElementById('download').href='/download?'+params()}
ids.forEach(id=>C[id].addEventListener('input',()=>{labels();clearTimeout(timer);timer=setTimeout(solve,170)}));document.getElementById('reset').onclick=()=>{Object.entries(D).forEach(([k,v])=>C[k].value=v);labels();solve()};
async function solve(){document.getElementById('statusTitle').textContent='Solving both circuits…';try{const r=await fetch('/api?'+params()),d=await r.json();if(!r.ok)throw Error(d.error||'Solver error');render(d)}catch(e){const s=document.getElementById('status');s.className='status invalid';document.getElementById('statusTitle').textContent='This operating point could not be solved';document.getElementById('statusDetail').textContent=e.message}}
function nodes(c){return Object.fromEntries(c.nodes.map(n=>[n.name,n]))}function edges(c){return Object.fromEntries(c.edges.map(e=>[e.name,e]))}
function render(d){const s=d.summary,L=d.lox,F=d.fuel,ln=nodes(L),fn=nodes(F);document.getElementById('loxFlow').textContent=s.lox_flow.toFixed(3);document.getElementById('fuelFlow').textContent=s.fuel_flow.toFixed(3);document.getElementById('totalFlow').textContent=s.total_flow.toFixed(3);document.getElementById('ofRatio').textContent=s.mixture_ratio.toFixed(2);document.getElementById('thrustMetric').textContent=s.estimated_thrust_kn.toFixed(2);document.getElementById('engineValue').textContent=s.chamber_pressure.toFixed(0)+' kPa';
for(const [id,n] of [['loxTank','Propellant tank'],['loxFeed','Feed manifold'],['loxValve','Valve outlet'],['loxInjector','Injector inlet']])document.getElementById(id).textContent=ln[n].pressure.toFixed(0)+' kPa';for(const [id,n] of [['fuelTank','Propellant tank'],['fuelFeed','Feed manifold'],['fuelValve','Valve outlet'],['fuelInjector','Injector inlet']])document.getElementById(id).textContent=fn[n].pressure.toFixed(0)+' kPa';document.getElementById('loxPipe').textContent=s.lox_flow.toFixed(2)+' kg/s';document.getElementById('fuelPipe').textContent=s.fuel_flow.toFixed(2)+' kg/s';
const status=document.getElementById('status');status.className='status '+s.operating_state;const titles={pass:'Design point satisfies the documented operating envelope',caution:'Solvable off-design point — review the highlighted constraints',invalid:'Physically invalid operating point'};document.getElementById('statusTitle').textContent=titles[s.operating_state];document.getElementById('statusDetail').textContent=s.issues.length?s.issues.join(' · '):`Flow is ${s.design_match_percent.toFixed(1)}% of design; O/F ${s.mixture_ratio.toFixed(2)}; injector drops ${(100*L.injector_drop_ratio).toFixed(1)}% and ${(100*F.injector_drop_ratio).toFixed(1)}%.`;document.getElementById('residual').textContent=`Residuals ${L.residual.toExponential(1)} / ${F.residual.toExponential(1)}`;
const lp=100*s.lox_flow/s.total_flow,fp=100-lp;document.getElementById('mixLox').style.width=lp+'%';document.getElementById('mixFuel').style.width=fp+'%';document.getElementById('splitText').textContent=`${lp.toFixed(0)}% LOX · ${fp.toFixed(0)}% fuel`;margin('loxMargin',L);margin('fuelMargin',F);drop('loxFeedDrop',L);drop('fuelFeedDrop',F);ratio('loxInjectorRatio',L);ratio('fuelInjectorRatio',F);drawChart(L.pressure_profile,F.pressure_profile);const losses=[...L.edges.map(e=>({...e,fluid:'LOX'})),...F.edges.map(e=>({...e,fluid:'LCH4'}))].filter(e=>e.loss>0).sort((a,b)=>b.loss-a.loss).slice(0,6);drawLosses(losses)}
function margin(id,c){const el=document.getElementById(id);el.textContent=c.minimum_saturation_margin.toFixed(0)+' kPa';el.className=c.minimum_saturation_margin>0?'pass-text':'fail-text'}
function drop(id,c){document.getElementById(id).textContent=c.feed_pressure_drop.toFixed(0)+' kPa'}
function ratio(id,c){const el=document.getElementById(id);el.textContent=(100*c.injector_drop_ratio).toFixed(1)+'%';el.className=c.injector_passes?'pass-text':'fail-text'}
function drawChart(A,B){const svg=document.getElementById('chart'),NS='http://www.w3.org/2000/svg',W=720,H=320,L=52,R=18,T=24,Bot=60,vals=[...A,...B].map(p=>p.pressure),mn=Math.min(...vals),mx=Math.max(...vals),pad=Math.max(30,(mx-mn)*.1),y0=mn-pad,y1=mx+pad,x=i=>L+i*(W-L-R)/(A.length-1),y=v=>T+(y1-v)*(H-T-Bot)/(y1-y0);svg.innerHTML='';for(let i=0;i<4;i++){let gy=T+i*(H-T-Bot)/3,line=document.createElementNS(NS,'line');line.setAttribute('x1',L);line.setAttribute('x2',W-R);line.setAttribute('y1',gy);line.setAttribute('y2',gy);line.setAttribute('class','grid');svg.append(line);let t=document.createElementNS(NS,'text');t.setAttribute('x',L-7);t.setAttribute('y',gy+3);t.setAttribute('text-anchor','end');t.setAttribute('class','chart-label');t.textContent=(y1-i*(y1-y0)/3).toFixed(0);svg.append(t)}for(const [data,cl,dc,dy] of [[A,'chart-lox','dot-lox',-11],[B,'chart-fuel','dot-fuel',15]]){let p=document.createElementNS(NS,'polyline');p.setAttribute('points',data.map((v,i)=>`${x(i)},${y(v.pressure)}`).join(' '));p.setAttribute('class',cl);svg.append(p);data.forEach((v,i)=>{let c=document.createElementNS(NS,'circle');c.setAttribute('cx',x(i));c.setAttribute('cy',y(v.pressure));c.setAttribute('r',4);c.setAttribute('class',dc);svg.append(c);let z=document.createElementNS(NS,'text');z.setAttribute('x',x(i));z.setAttribute('y',y(v.pressure)+dy);z.setAttribute('class','chart-value');z.textContent=v.pressure.toFixed(0);svg.append(z);if(cl==='chart-lox'){let q=document.createElementNS(NS,'text');q.setAttribute('x',x(i));q.setAttribute('y',H-Bot+20);q.setAttribute('class','chart-label');q.textContent=v.short;svg.append(q)}})}}
function drawLosses(a){const m=Math.max(...a.map(e=>e.loss));document.getElementById('lossBars').innerHTML=a.map(e=>`<div class="loss-row"><span class="loss-name">${e.fluid} · ${e.name}</span><div class="track"><div class="fill" style="width:${100*e.loss/m}%"></div></div><span class="loss-value">${e.loss.toFixed(1)}</span></div>`).join('')}
labels();solve();
</script></body></html>'''


def _bounded_query(query: dict[str, list[str]], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(query.get(key, [str(default)])[0])
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def dashboard_payload(query: dict[str, list[str]]) -> dict[str, object]:
    source_pressure_kpa = _bounded_query(query, "source_pressure", 450.0, 150.0, 800.0)
    temperature_k = _bounded_query(query, "temperature", 90.0, 87.0, 100.0)
    valve_percent = _bounded_query(query, "valve", 100.0, 20.0, 100.0)
    pump_percent = _bounded_query(query, "pump", 100.0, 50.0, 110.0)
    receiver_pressure_kpa = _bounded_query(query, "receiver_pressure", 300.0, 200.0, 600.0)
    scenario = Scenario("Interactive", main_valve_opening=valve_percent / 100.0, pump_health=pump_percent / 100.0, source_pressure_pa=source_pressure_kpa * 1000.0, source_temperature_k=temperature_k, receiver_pressure_pa=receiver_pressure_kpa * 1000.0)
    network, solution = solve_scenario(scenario)
    pump = network.pump_check(solution)
    nodes = [{"name": name, "pressure": pressure / 1000.0, "temperature": solution.node_temperatures_k[name], "saturation_margin": solution.saturation_margin_pa(name) / 1000.0} for name, pressure in solution.node_pressures_pa.items()]
    edges = [{"name": item.name, "start": item.start, "end": item.end, "flow": item.mass_flow_kg_s, "loss": item.pressure_loss_pa / 1000.0, "gain": item.pressure_gain_pa / 1000.0} for item in solution.edge_results.values()]
    pressure_path = [("Storage tank", "Tank"), ("Pump inlet", "Pump in"), ("Pump outlet", "Pump out"), ("Main branch", "Main"), ("Branch merge", "Merge"), ("Coupler outlet", "Coupler"), ("Vehicle tank", "Vehicle")]
    return {"summary": {"receiver_flow": solution.flow_into("Vehicle tank"), "pump_inlet": solution.node_pressures_pa["Pump inlet"] / 1000.0, "pump_outlet": solution.node_pressures_pa["Pump outlet"] / 1000.0, "npsh_available": float(pump["available_m"]), "npsh_required": float(pump["required_m"]), "npsh_margin": float(pump["margin_m"]), "passes": bool(pump["passes"]), "residual": solution.residual_norm}, "nodes": nodes, "edges": edges, "pressure_profile": [{"name": name, "short": short, "pressure": solution.node_pressures_pa[name] / 1000.0} for name, short in pressure_path]}


def payload_csv(payload: dict[str, object]) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(["CRYONET INTERACTIVE RESULT"])
    writer.writerow(["metric", "value", "unit"])
    summary = payload["summary"]
    writer.writerow(["receiver_flow", summary["receiver_flow"], "kg/s"])
    writer.writerow(["pump_inlet_pressure", summary["pump_inlet"], "kPa abs"])
    writer.writerow(["pump_outlet_pressure", summary["pump_outlet"], "kPa abs"])
    writer.writerow(["npsh_margin", summary["npsh_margin"], "m"])
    writer.writerow([])
    writer.writerow(["node", "pressure_kpa_abs", "temperature_k", "saturation_margin_kpa"])
    for node in payload["nodes"]:
        writer.writerow([node["name"], node["pressure"], node["temperature"], node["saturation_margin"]])
    writer.writerow([])
    writer.writerow(["component", "start", "end", "flow_kg_s", "loss_kpa", "gain_kpa"])
    for edge in payload["edges"]:
        writer.writerow([edge["name"], edge["start"], edge["end"], edge["flow"], edge["loss"], edge["gain"]])
    return handle.getvalue()


def _feed_payload(network: Network, solution: Solution) -> dict[str, object]:
    """Convert one solved circuit into browser-friendly engineering units."""
    del network
    nodes = [
        {
            "name": name,
            "pressure": pressure / 1000.0,
            "temperature": solution.node_temperatures_k[name],
            "saturation_margin": solution.saturation_margin_pa(name) / 1000.0,
        }
        for name, pressure in solution.node_pressures_pa.items()
    ]
    edges = [
        {
            "name": item.name,
            "start": item.start,
            "end": item.end,
            "flow": item.mass_flow_kg_s,
            "loss": item.pressure_loss_pa / 1000.0,
            "gain": item.pressure_gain_pa / 1000.0,
        }
        for item in solution.edge_results.values()
    ]
    path = [
        ("Propellant tank", "Tank"),
        ("Feed manifold", "Manifold"),
        ("Valve outlet", "Valve"),
        ("Injector inlet", "Inj. in"),
        ("Combustion chamber", "Chamber"),
    ]
    chamber_pressure = solution.node_pressures_pa["Combustion chamber"]
    injector_drop = (
        solution.node_pressures_pa["Injector inlet"] - chamber_pressure
    )
    injector_ratio = injector_drop / chamber_pressure
    injector_edge = solution.edge_results["Injector orifice"]
    injector_low, injector_high = MORPHEUS.injector_band(chamber_pressure)
    liquid_nodes = [name for name, _ in path[:-1]]
    minimum_saturation_margin = min(
        solution.saturation_margin_pa(name) for name in liquid_nodes
    )
    pressures = [solution.node_pressures_pa[name] for name, _ in path]
    pressure_ordering = all(a > b for a, b in zip(pressures, pressures[1:]))
    return {
        "fluid": solution.fluid.name,
        "flow": solution.flow_into("Combustion chamber"),
        "feed_pressure_drop": (
            solution.node_pressures_pa["Propellant tank"]
            - solution.node_pressures_pa["Injector inlet"]
        )
        / 1000.0,
        "minimum_saturation_margin": minimum_saturation_margin / 1000.0,
        "pressure_ordering": pressure_ordering,
        "passes": pressure_ordering and minimum_saturation_margin > 0.0,
        "injector_pressure_drop": injector_drop / 1000.0,
        "injector_drop_ratio": injector_ratio,
        "injector_band_min": injector_low,
        "injector_band_max": injector_high,
        "injector_passes": injector_low <= injector_ratio <= injector_high,
        "injector_area_mm2": injector_edge.metrics["total_flow_area_mm2"],
        "injector_cd": injector_edge.metrics["discharge_coefficient"],
        "residual": solution.residual_norm,
        "nodes": nodes,
        "edges": edges,
        "pressure_profile": [
            {
                "name": name,
                "short": short,
                "pressure": solution.node_pressures_pa[name] / 1000.0,
            }
            for name, short in path
        ],
    }


def propulsion_dashboard_payload(query: dict[str, list[str]]) -> dict[str, object]:
    """Solve the dashboard operating point from bounded URL parameters."""
    valve_opening = _bounded_query(query, "valve", 82.0, 35.0, 95.0) / 100.0
    scenario = PropulsionScenario(
        name="Interactive",
        lox_tank_pressure_pa=_bounded_query(
            query, "lox_pressure", MORPHEUS.design_tank_pressure_pa / 1000.0, 2100.0, 2700.0
        )
        * 1000.0,
        methane_tank_pressure_pa=_bounded_query(
            query, "fuel_pressure", MORPHEUS.design_tank_pressure_pa / 1000.0, 2100.0, 2700.0
        )
        * 1000.0,
        lox_temperature_k=_bounded_query(query, "lox_temperature", 90.0, 84.0, 105.0),
        methane_temperature_k=_bounded_query(
            query, "fuel_temperature", 112.0, 100.0, 128.0
        ),
        chamber_pressure_pa=_bounded_query(
            query, "chamber_pressure", MORPHEUS.design_chamber_pressure_pa / 1000.0, 550.0, 1800.0
        )
        * 1000.0,
        lox_valve_opening=valve_opening,
        methane_valve_opening=valve_opening,
        target_mixture_ratio=MORPHEUS.target_mixture_ratio,
    )
    result = solve_propulsion_system(scenario)
    injector_low, injector_high = MORPHEUS.injector_band(
        scenario.chamber_pressure_pa
    )
    issues = result.physical_issues() + result.requirement_issues()
    return {
        "summary": {
            "lox_flow": result.lox_flow_kg_s,
            "fuel_flow": result.methane_flow_kg_s,
            "total_flow": result.total_flow_kg_s,
            "mixture_ratio": result.mixture_ratio,
            "target_mixture_ratio": scenario.target_mixture_ratio,
            "mixture_error": result.mixture_error,
            "mixture_passes": result.mixture_passes,
            "lox_injector_passes": result.lox_injector_passes,
            "fuel_injector_passes": result.methane_injector_passes,
            "injector_ratio_min": injector_low,
            "injector_ratio_max": injector_high,
            "system_passes": result.system_passes,
            "operating_state": result.operating_state,
            "issues": issues,
            "chamber_pressure": scenario.chamber_pressure_pa / 1000.0,
            "estimated_thrust_kn": result.estimated_thrust_n / 1000.0,
            "design_thrust_kn": MORPHEUS.design_thrust_n / 1000.0,
            "design_total_flow": MORPHEUS.design_total_flow_kg_s,
            "design_match_percent": 100.0
            * result.total_flow_kg_s
            / MORPHEUS.design_total_flow_kg_s,
        },
        "lox": _feed_payload(result.lox_network, result.lox),
        "fuel": _feed_payload(result.methane_network, result.methane),
        "model_note": (
            "Morpheus HD4-A requirements-based, pressure-fed, steady-state, "
            "single-phase reconstruction. Published requirements are separated "
            "from calibrated equivalent hardware and identified assumptions."
        ),
        "design_basis": {
            "engine": MORPHEUS.engine_name,
            "architecture": MORPHEUS.architecture,
            "thrust_lbf": 4200.0,
            "specific_impulse_s": MORPHEUS.specific_impulse_s,
            "throttle_ratio": MORPHEUS.throttle_ratio,
            "tank_pressure_psia": MORPHEUS.design_tank_pressure_pa / PSI_TO_PA,
            "engine_inlet_psia": MORPHEUS.minimum_engine_inlet_pressure_pa / PSI_TO_PA,
            "chamber_pressure_psia": MORPHEUS.design_chamber_pressure_pa / PSI_TO_PA,
            "target_of": MORPHEUS.target_mixture_ratio,
        },
    }


def propulsion_payload_csv(payload: dict[str, object]) -> str:
    """Create one downloadable report containing both propellant circuits."""
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(["CRYONET DUAL-PROPELLANT ENGINE-FEED RESULT"])
    writer.writerow(["metric", "value", "unit"])
    summary = payload["summary"]
    for key, unit in (
        ("lox_flow", "kg/s"),
        ("fuel_flow", "kg/s"),
        ("total_flow", "kg/s"),
        ("mixture_ratio", "O/F"),
        ("target_mixture_ratio", "O/F"),
        ("chamber_pressure", "kPa abs"),
        ("estimated_thrust_kn", "kN"),
        ("design_match_percent", "% of design flow"),
    ):
        writer.writerow([key, summary[key], unit])
    writer.writerow(["operating_state", summary["operating_state"], ""])
    writer.writerow(["issues", " | ".join(summary["issues"]), ""])
    writer.writerow(
        [
            "active_injector_screen",
            f"{100*summary['injector_ratio_min']:.1f} to {100*summary['injector_ratio_max']:.1f}",
            "% of chamber pressure",
        ]
    )
    writer.writerow([])
    writer.writerow(["DESIGN BASIS"])
    for key, value in payload["design_basis"].items():
        writer.writerow([key, value])
    for circuit_name in ("lox", "fuel"):
        circuit = payload[circuit_name]
        writer.writerow([])
        writer.writerow([circuit_name.upper(), circuit["fluid"]])
        writer.writerow(
            [
                "injector_pressure_drop_ratio",
                100.0 * circuit["injector_drop_ratio"],
                "% of chamber pressure",
            ]
        )
        writer.writerow(["feed_pressure_drop", circuit["feed_pressure_drop"], "kPa"])
        writer.writerow(
            [
                "minimum_liquid_saturation_margin",
                circuit["minimum_saturation_margin"],
                "kPa",
            ]
        )
        writer.writerow(
            ["node", "pressure_kpa_abs", "temperature_k", "saturation_margin_kpa"]
        )
        for node in circuit["nodes"]:
            writer.writerow(
                [
                    node["name"],
                    node["pressure"],
                    node["temperature"],
                    node["saturation_margin"],
                ]
            )
        writer.writerow([])
        writer.writerow(
            ["component", "start", "end", "flow_kg_s", "loss_kpa", "gain_kpa"]
        )
        for edge in circuit["edges"]:
            writer.writerow(
                [
                    edge["name"],
                    edge["start"],
                    edge["end"],
                    edge["flow"],
                    edge["loss"],
                    edge["gain"],
                ]
            )
    writer.writerow([])
    writer.writerow(["MODEL NOTE", payload["model_note"]])
    return handle.getvalue()


class CryoNetHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            content = DUAL_DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif parsed.path == "/api":
            try:
                content = json.dumps(propulsion_dashboard_payload(query)).encode("utf-8")
                self.send_response(200)
            except Exception as error:
                content = json.dumps({"error": str(error)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif parsed.path == "/download":
            content = propulsion_payload_csv(
                propulsion_dashboard_payload(query)
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", "attachment; filename=cryonet_dual_feed_result.csv"
            )
        else:
            content = b"Not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def launch_dashboard() -> None:
    run_self_checks()
    port = 8765
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), CryoNetHandler)
            break
        except OSError:
            port += 1
            if port > 8790:
                raise RuntimeError("Could not find an open local port.")
    url = f"http://127.0.0.1:{port}"
    print("\nCryoNet Interactive is running.")
    print(f"Open: {url}")
    print("Press Control-C in this terminal to stop the dashboard.\n")
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCryoNet dashboard stopped.")
    finally:
        server.server_close()


def main() -> None:
    if "--export" in sys.argv:
        export_static_results()
    else:
        launch_dashboard()


if __name__ == "__main__":
    main()
