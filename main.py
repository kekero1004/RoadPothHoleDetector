# -*- coding: utf-8 -*-
"""
Road Pothole Detector - Main Plugin Module
AI-based Road Pothole/Crack Detection & Maintenance Route Optimization

This module contains the main plugin class with GUI and business logic.
"""

import os
import math
import random
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from qgis.PyQt.QtCore import Qt, QVariant, QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon, QColor, QFont, QPixmap
from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QSizePolicy, QApplication, QStyle
)

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsField, QgsFields, QgsWkbTypes,
    QgsCoordinateReferenceSystem, QgsSymbol, QgsMarkerSymbol,
    QgsCategorizedSymbolRenderer, QgsRendererCategory,
    QgsVectorFileWriter, QgsMessageLog, Qgis,
    QgsSingleSymbolRenderer, QgsLineSymbol
)
from qgis.gui import QgsMapCanvas


# Base64 encoded icon will be loaded from icon.png file
PLUGIN_NAME = "Road Pothole Detector"


class DefectData:
    """Class to represent a detected road defect."""

    def __init__(self, defect_id: int, x: float, y: float, defect_type: str,
                 severity: str, confidence: float, image_path: str = ""):
        self.id = defect_id
        self.x = x
        self.y = y
        self.defect_type = defect_type  # pothole, alligator_crack, linear_crack
        self.severity = severity  # A, B, C, D, E (A=minor, E=critical)
        self.confidence = confidence
        self.image_path = image_path
        self.estimated_cost = self._calculate_cost()
        self.priority = self._calculate_priority()

    def _calculate_cost(self) -> float:
        """Calculate estimated repair cost based on defect type and severity."""
        base_costs = {
            "pothole": 50000,
            "alligator_crack": 80000,
            "linear_crack": 30000
        }
        severity_multipliers = {"A": 0.5, "B": 0.75, "C": 1.0, "D": 1.5, "E": 2.0}

        base = base_costs.get(self.defect_type, 50000)
        multiplier = severity_multipliers.get(self.severity, 1.0)
        return base * multiplier

    def _calculate_priority(self) -> int:
        """Calculate maintenance priority (1=highest, 5=lowest)."""
        priority_map = {"E": 1, "D": 2, "C": 3, "B": 4, "A": 5}
        return priority_map.get(self.severity, 3)


class TSPSolver:
    """Simple TSP solver using nearest neighbor heuristic."""

    @staticmethod
    def solve(points: List[Tuple[float, float]], start_index: int = 0) -> List[int]:
        """
        Solve TSP using nearest neighbor algorithm.

        :param points: List of (x, y) coordinates
        :param start_index: Starting point index
        :returns: Ordered list of point indices
        """
        if len(points) <= 1:
            return list(range(len(points)))

        n = len(points)
        visited = [False] * n
        route = [start_index]
        visited[start_index] = True

        for _ in range(n - 1):
            current = route[-1]
            nearest = -1
            min_dist = float('inf')

            for j in range(n):
                if not visited[j]:
                    dist = TSPSolver._distance(points[current], points[j])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j

            if nearest != -1:
                route.append(nearest)
                visited[nearest] = True

        return route

    @staticmethod
    def solve_with_priority(defects: List[DefectData], depot: Tuple[float, float] = None) -> List[int]:
        """
        Solve TSP considering defect priority (D, E grades first).

        :param defects: List of DefectData objects
        :param depot: Starting depot coordinates
        :returns: Ordered list of defect indices
        """
        if not defects:
            return []

        # Separate high priority (D, E) and lower priority defects
        high_priority = [(i, d) for i, d in enumerate(defects) if d.severity in ["D", "E"]]
        low_priority = [(i, d) for i, d in enumerate(defects) if d.severity not in ["D", "E"]]

        # Sort each group by priority
        high_priority.sort(key=lambda x: x[1].priority)
        low_priority.sort(key=lambda x: x[1].priority)

        # Solve TSP for high priority first
        high_points = [(d.x, d.y) for _, d in high_priority]
        high_indices = [i for i, _ in high_priority]

        if high_points:
            high_route = TSPSolver.solve(high_points)
            ordered_high = [high_indices[i] for i in high_route]
        else:
            ordered_high = []

        # Then low priority
        low_points = [(d.x, d.y) for _, d in low_priority]
        low_indices = [i for i, _ in low_priority]

        if low_points:
            low_route = TSPSolver.solve(low_points)
            ordered_low = [low_indices[i] for i in low_route]
        else:
            ordered_low = []

        return ordered_high + ordered_low

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def calculate_total_distance(points: List[Tuple[float, float]], route: List[int]) -> float:
        """Calculate total route distance."""
        if len(route) < 2:
            return 0.0

        total = 0.0
        for i in range(len(route) - 1):
            total += TSPSolver._distance(points[route[i]], points[route[i+1]])
        return total


class LCCAnalyzer:
    """Life Cycle Cost Analyzer for maintenance decision support."""

    INFLATION_RATE = 0.03
    DISCOUNT_RATE = 0.05

    @staticmethod
    def analyze(defects: List[DefectData], years: int = 5) -> Dict:
        """
        Analyze Life Cycle Cost for immediate vs delayed repair scenarios.

        :param defects: List of defect data
        :param years: Analysis period in years
        :returns: Analysis results dictionary
        """
        immediate_cost = sum(d.estimated_cost for d in defects)

        # Delayed repair scenario - costs increase over time
        delayed_costs = []
        cumulative = 0
        severity_growth = {"A": 1.1, "B": 1.2, "C": 1.3, "D": 1.5, "E": 1.8}

        for year in range(years):
            year_cost = 0
            for d in defects:
                growth = severity_growth.get(d.severity, 1.2)
                adjusted = d.estimated_cost * (growth ** year)
                year_cost += adjusted * ((1 + LCCAnalyzer.INFLATION_RATE) ** year)

            # Apply discount rate for NPV
            npv_cost = year_cost / ((1 + LCCAnalyzer.DISCOUNT_RATE) ** year)
            cumulative += npv_cost
            delayed_costs.append(cumulative)

        # Calculate savings
        final_delayed = delayed_costs[-1] if delayed_costs else 0
        savings = final_delayed - immediate_cost
        savings_percent = (savings / final_delayed * 100) if final_delayed > 0 else 0

        return {
            "immediate_cost": immediate_cost,
            "delayed_costs": delayed_costs,
            "total_delayed_cost": final_delayed,
            "savings": savings,
            "savings_percent": savings_percent,
            "recommendation": "Immediate repair recommended" if savings > 0 else "Delayed repair may be acceptable"
        }


class DetectionSimulator:
    """Simulates AI detection for demonstration purposes."""

    DEFECT_TYPES = ["pothole", "alligator_crack", "linear_crack"]
    SEVERITIES = ["A", "B", "C", "D", "E"]

    @staticmethod
    def simulate_detection(center_x: float, center_y: float,
                          radius: float, count: int) -> List[DefectData]:
        """
        Simulate defect detection around a center point.

        :param center_x: Center X coordinate
        :param center_y: Center Y coordinate
        :param radius: Spread radius
        :param count: Number of defects to generate
        :returns: List of simulated DefectData
        """
        defects = []

        for i in range(count):
            # Random position within radius
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, radius)
            x = center_x + dist * math.cos(angle)
            y = center_y + dist * math.sin(angle)

            # Random defect properties
            defect_type = random.choice(DetectionSimulator.DEFECT_TYPES)
            severity = random.choices(
                DetectionSimulator.SEVERITIES,
                weights=[0.15, 0.25, 0.30, 0.20, 0.10]  # More medium severity
            )[0]
            confidence = random.uniform(0.75, 0.99)

            defect = DefectData(
                defect_id=i + 1,
                x=x,
                y=y,
                defect_type=defect_type,
                severity=severity,
                confidence=confidence
            )
            defects.append(defect)

        return defects


class MainDialog(QDialog):
    """Main dialog window for Road Pothole Detector plugin."""

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.defects: List[DefectData] = []
        self.route_order: List[int] = []
        self.defect_layer = None
        self.route_layer = None

        self.setWindowTitle(PLUGIN_NAME)
        self.setMinimumSize(900, 700)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        main_layout = QVBoxLayout(self)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Tab widget for main functionality
        self.tabs = QTabWidget()

        # Tab 1: Detection
        detection_tab = self._create_detection_tab()
        self.tabs.addTab(detection_tab, "1. Detection")

        # Tab 2: Risk Assessment
        assessment_tab = self._create_assessment_tab()
        self.tabs.addTab(assessment_tab, "2. Risk Assessment")

        # Tab 3: Route Optimization
        route_tab = self._create_route_tab()
        self.tabs.addTab(route_tab, "3. Route Optimization")

        # Tab 4: LCC Analysis
        lcc_tab = self._create_lcc_tab()
        self.tabs.addTab(lcc_tab, "4. LCC Analysis")

        # Tab 5: Report Generation
        report_tab = self._create_report_tab()
        self.tabs.addTab(report_tab, "5. Report")

        main_layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #666; padding: 5px;")
        main_layout.addWidget(self.status_bar)

    def _create_header(self) -> QWidget:
        """Create header section."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                color: white;
            }
        """)

        layout = QHBoxLayout(header)

        title = QLabel(f"<h2>{PLUGIN_NAME}</h2>")
        title.setStyleSheet("font-weight: bold; color: white;")
        layout.addWidget(title)

        layout.addStretch()

        subtitle = QLabel("AI-based Detection & Route Optimization")
        subtitle.setStyleSheet("color: #bdc3c7;")
        layout.addWidget(subtitle)

        return header

    def _create_detection_tab(self) -> QWidget:
        """Create detection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input section
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout(input_group)

        # Image/Video input
        input_layout.addWidget(QLabel("Image/Video File:"), 0, 0)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select drone/vehicle footage...")
        input_layout.addWidget(self.image_path, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_image)
        input_layout.addWidget(browse_btn, 0, 2)

        # Center coordinates for simulation
        input_layout.addWidget(QLabel("Center X (Lon):"), 1, 0)
        self.center_x = QDoubleSpinBox()
        self.center_x.setRange(-180, 180)
        self.center_x.setDecimals(6)
        self.center_x.setValue(127.0)  # Default: Seoul area
        input_layout.addWidget(self.center_x, 1, 1)

        input_layout.addWidget(QLabel("Center Y (Lat):"), 2, 0)
        self.center_y = QDoubleSpinBox()
        self.center_y.setRange(-90, 90)
        self.center_y.setDecimals(6)
        self.center_y.setValue(37.5)  # Default: Seoul area
        input_layout.addWidget(self.center_y, 2, 1)

        input_layout.addWidget(QLabel("Detection Radius (m):"), 3, 0)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(10, 10000)
        self.radius.setValue(500)
        input_layout.addWidget(self.radius, 3, 1)

        input_layout.addWidget(QLabel("Simulated Defects:"), 4, 0)
        self.defect_count = QSpinBox()
        self.defect_count.setRange(1, 500)
        self.defect_count.setValue(25)
        input_layout.addWidget(self.defect_count, 4, 1)

        layout.addWidget(input_group)

        # Detection options
        options_group = QGroupBox("Detection Options")
        options_layout = QGridLayout(options_group)

        options_layout.addWidget(QLabel("Detection Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv7 (Simulated)", "YOLOv8 (Simulated)", "Custom Model"])
        options_layout.addWidget(self.model_combo, 0, 1)

        options_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_thresh = QDoubleSpinBox()
        self.confidence_thresh.setRange(0.1, 1.0)
        self.confidence_thresh.setSingleStep(0.05)
        self.confidence_thresh.setValue(0.75)
        options_layout.addWidget(self.confidence_thresh, 1, 1)

        self.detect_potholes = QCheckBox("Detect Potholes")
        self.detect_potholes.setChecked(True)
        options_layout.addWidget(self.detect_potholes, 2, 0)

        self.detect_cracks = QCheckBox("Detect Cracks")
        self.detect_cracks.setChecked(True)
        options_layout.addWidget(self.detect_cracks, 2, 1)

        layout.addWidget(options_group)

        # Progress bar
        self.detection_progress = QProgressBar()
        self.detection_progress.setVisible(False)
        layout.addWidget(self.detection_progress)

        # Run detection button
        detect_btn = QPushButton("Run Detection")
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        detect_btn.clicked.connect(self._run_detection)
        layout.addWidget(detect_btn)

        # Results summary
        self.detection_result = QTextEdit()
        self.detection_result.setReadOnly(True)
        self.detection_result.setMaximumHeight(150)
        layout.addWidget(self.detection_result)

        layout.addStretch()
        return tab

    def _create_assessment_tab(self) -> QWidget:
        """Create risk assessment tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics group
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout(stats_group)

        self.stat_labels = {}
        stats = [
            ("total", "Total Defects:"),
            ("potholes", "Potholes:"),
            ("alligator", "Alligator Cracks:"),
            ("linear", "Linear Cracks:"),
            ("grade_a", "Grade A (Minor):"),
            ("grade_b", "Grade B:"),
            ("grade_c", "Grade C:"),
            ("grade_d", "Grade D:"),
            ("grade_e", "Grade E (Critical):")
        ]

        for i, (key, label) in enumerate(stats):
            stats_layout.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            self.stat_labels[key] = QLabel("0")
            self.stat_labels[key].setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(self.stat_labels[key], i // 2, (i % 2) * 2 + 1)

        layout.addWidget(stats_group)

        # Defects table
        table_group = QGroupBox("Detected Defects")
        table_layout = QVBoxLayout(table_group)

        self.defects_table = QTableWidget()
        self.defects_table.setColumnCount(7)
        self.defects_table.setHorizontalHeaderLabels([
            "ID", "Type", "Severity", "Confidence", "X", "Y", "Est. Cost"
        ])
        self.defects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.defects_table.setSelectionBehavior(QTableWidget.SelectRows)
        table_layout.addWidget(self.defects_table)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by Severity:"))
        self.severity_filter = QComboBox()
        self.severity_filter.addItems(["All", "A", "B", "C", "D", "E"])
        self.severity_filter.currentTextChanged.connect(self._filter_defects)
        filter_layout.addWidget(self.severity_filter)

        filter_layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "pothole", "alligator_crack", "linear_crack"])
        self.type_filter.currentTextChanged.connect(self._filter_defects)
        filter_layout.addWidget(self.type_filter)

        filter_layout.addStretch()

        zoom_btn = QPushButton("Zoom to Selected")
        zoom_btn.clicked.connect(self._zoom_to_selected)
        filter_layout.addWidget(zoom_btn)

        table_layout.addLayout(filter_layout)
        layout.addWidget(table_group)

        return tab

    def _create_route_tab(self) -> QWidget:
        """Create route optimization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Route options
        options_group = QGroupBox("Route Options")
        options_layout = QGridLayout(options_group)

        options_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Nearest Neighbor (Fast)",
            "Priority-Based TSP",
            "2-Opt Improvement"
        ])
        options_layout.addWidget(self.algorithm_combo, 0, 1)

        options_layout.addWidget(QLabel("Depot X:"), 1, 0)
        self.depot_x = QDoubleSpinBox()
        self.depot_x.setRange(-180, 180)
        self.depot_x.setDecimals(6)
        self.depot_x.setValue(127.0)
        options_layout.addWidget(self.depot_x, 1, 1)

        options_layout.addWidget(QLabel("Depot Y:"), 2, 0)
        self.depot_y = QDoubleSpinBox()
        self.depot_y.setRange(-90, 90)
        self.depot_y.setDecimals(6)
        self.depot_y.setValue(37.5)
        options_layout.addWidget(self.depot_y, 2, 1)

        self.priority_first = QCheckBox("Visit high-priority (D, E) defects first")
        self.priority_first.setChecked(True)
        options_layout.addWidget(self.priority_first, 3, 0, 1, 2)

        layout.addWidget(options_group)

        # Generate route button
        route_btn = QPushButton("Generate Optimal Route")
        route_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        route_btn.clicked.connect(self._generate_route)
        layout.addWidget(route_btn)

        # Route results
        result_group = QGroupBox("Route Results")
        result_layout = QVBoxLayout(result_group)

        self.route_result = QTextEdit()
        self.route_result.setReadOnly(True)
        result_layout.addWidget(self.route_result)

        # Export buttons
        export_layout = QHBoxLayout()

        export_gpx_btn = QPushButton("Export GPX")
        export_gpx_btn.clicked.connect(self._export_gpx)
        export_layout.addWidget(export_gpx_btn)

        export_shp_btn = QPushButton("Export Shapefile")
        export_shp_btn.clicked.connect(self._export_shapefile)
        export_layout.addWidget(export_shp_btn)

        result_layout.addLayout(export_layout)
        layout.addWidget(result_group)

        layout.addStretch()
        return tab

    def _create_lcc_tab(self) -> QWidget:
        """Create LCC analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Analysis options
        options_group = QGroupBox("Analysis Parameters")
        options_layout = QGridLayout(options_group)

        options_layout.addWidget(QLabel("Analysis Period (years):"), 0, 0)
        self.lcc_years = QSpinBox()
        self.lcc_years.setRange(1, 20)
        self.lcc_years.setValue(5)
        options_layout.addWidget(self.lcc_years, 0, 1)

        options_layout.addWidget(QLabel("Inflation Rate (%):"), 1, 0)
        self.inflation_rate = QDoubleSpinBox()
        self.inflation_rate.setRange(0, 20)
        self.inflation_rate.setValue(3.0)
        options_layout.addWidget(self.inflation_rate, 1, 1)

        options_layout.addWidget(QLabel("Discount Rate (%):"), 2, 0)
        self.discount_rate = QDoubleSpinBox()
        self.discount_rate.setRange(0, 20)
        self.discount_rate.setValue(5.0)
        options_layout.addWidget(self.discount_rate, 2, 1)

        layout.addWidget(options_group)

        # Run analysis button
        analyze_btn = QPushButton("Run LCC Analysis")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        analyze_btn.clicked.connect(self._run_lcc_analysis)
        layout.addWidget(analyze_btn)

        # Results
        result_group = QGroupBox("Analysis Results")
        result_layout = QVBoxLayout(result_group)

        self.lcc_result = QTextEdit()
        self.lcc_result.setReadOnly(True)
        result_layout.addWidget(self.lcc_result)

        layout.addWidget(result_group)

        layout.addStretch()
        return tab

    def _create_report_tab(self) -> QWidget:
        """Create report generation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Report options
        options_group = QGroupBox("Report Options")
        options_layout = QVBoxLayout(options_group)

        self.include_map = QCheckBox("Include Map Screenshot")
        self.include_map.setChecked(True)
        options_layout.addWidget(self.include_map)

        self.include_stats = QCheckBox("Include Detection Statistics")
        self.include_stats.setChecked(True)
        options_layout.addWidget(self.include_stats)

        self.include_route = QCheckBox("Include Route Information")
        self.include_route.setChecked(True)
        options_layout.addWidget(self.include_route)

        self.include_lcc = QCheckBox("Include LCC Analysis")
        self.include_lcc.setChecked(True)
        options_layout.addWidget(self.include_lcc)

        self.include_defect_list = QCheckBox("Include Defect List")
        self.include_defect_list.setChecked(True)
        options_layout.addWidget(self.include_defect_list)

        layout.addWidget(options_group)

        # Output path
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout(path_group)

        path_layout.addWidget(QLabel("Save to:"))
        self.report_path = QLineEdit()
        self.report_path.setPlaceholderText("Select output path...")
        path_layout.addWidget(self.report_path)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_report_path)
        path_layout.addWidget(browse_btn)

        layout.addWidget(path_group)

        # Generate button
        generate_btn = QPushButton("Generate Work Order Report")
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        generate_btn.clicked.connect(self._generate_report)
        layout.addWidget(generate_btn)

        # Preview
        preview_group = QGroupBox("Report Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        preview_layout.addWidget(self.report_preview)

        layout.addWidget(preview_group)

        return tab

    def _browse_image(self):
        """Open file dialog for image/video selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image or Video File",
            "",
            "Images/Videos (*.jpg *.jpeg *.png *.mp4 *.avi *.mov);;All Files (*.*)"
        )
        if file_path:
            self.image_path.setText(file_path)

    def _browse_report_path(self):
        """Open file dialog for report save location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "",
            "Text Files (*.txt);;HTML Files (*.html);;All Files (*.*)"
        )
        if file_path:
            self.report_path.setText(file_path)

    def _run_detection(self):
        """Run defect detection (simulated)."""
        self.detection_progress.setVisible(True)
        self.detection_progress.setValue(0)
        self.status_bar.setText("Running detection...")
        QApplication.processEvents()

        # Simulate detection progress
        for i in range(101):
            self.detection_progress.setValue(i)
            QApplication.processEvents()

        # Generate simulated defects
        center_x = self.center_x.value()
        center_y = self.center_y.value()
        radius = self.radius.value() / 111000  # Convert meters to degrees (approximate)
        count = self.defect_count.value()

        self.defects = DetectionSimulator.simulate_detection(
            center_x, center_y, radius, count
        )

        # Filter by confidence threshold
        threshold = self.confidence_thresh.value()
        self.defects = [d for d in self.defects if d.confidence >= threshold]

        # Create QGIS layer
        self._create_defect_layer()

        # Update UI
        self._update_statistics()
        self._populate_defects_table()

        # Show results
        result_text = f"""Detection Complete!

Model: {self.model_combo.currentText()}
Total Defects Found: {len(self.defects)}
Confidence Threshold: {threshold:.2f}

Defects by Type:
- Potholes: {sum(1 for d in self.defects if d.defect_type == 'pothole')}
- Alligator Cracks: {sum(1 for d in self.defects if d.defect_type == 'alligator_crack')}
- Linear Cracks: {sum(1 for d in self.defects if d.defect_type == 'linear_crack')}

Defects by Severity:
- Grade A (Minor): {sum(1 for d in self.defects if d.severity == 'A')}
- Grade B: {sum(1 for d in self.defects if d.severity == 'B')}
- Grade C: {sum(1 for d in self.defects if d.severity == 'C')}
- Grade D: {sum(1 for d in self.defects if d.severity == 'D')}
- Grade E (Critical): {sum(1 for d in self.defects if d.severity == 'E')}

Total Estimated Repair Cost: {sum(d.estimated_cost for d in self.defects):,.0f} KRW
"""
        self.detection_result.setText(result_text)

        self.detection_progress.setVisible(False)
        self.status_bar.setText(f"Detection complete. {len(self.defects)} defects found.")

        # Switch to assessment tab
        self.tabs.setCurrentIndex(1)

    def _create_defect_layer(self):
        """Create QGIS vector layer for defects."""
        # Remove existing layer if present
        if self.defect_layer:
            QgsProject.instance().removeMapLayer(self.defect_layer.id())

        # Create new memory layer
        self.defect_layer = QgsVectorLayer(
            "Point?crs=EPSG:4326",
            "Road Defects",
            "memory"
        )

        provider = self.defect_layer.dataProvider()

        # Add fields
        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("type", QVariant.String))
        fields.append(QgsField("severity", QVariant.String))
        fields.append(QgsField("confidence", QVariant.Double))
        fields.append(QgsField("priority", QVariant.Int))
        fields.append(QgsField("est_cost", QVariant.Double))
        provider.addAttributes(fields)
        self.defect_layer.updateFields()

        # Add features
        features = []
        for defect in self.defects:
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(defect.x, defect.y)))
            feature.setAttributes([
                defect.id,
                defect.defect_type,
                defect.severity,
                defect.confidence,
                defect.priority,
                defect.estimated_cost
            ])
            features.append(feature)

        provider.addFeatures(features)

        # Style by severity
        categories = []
        severity_colors = {
            "A": ("#27ae60", "Grade A - Minor"),
            "B": ("#f1c40f", "Grade B"),
            "C": ("#e67e22", "Grade C"),
            "D": ("#e74c3c", "Grade D"),
            "E": ("#8e44ad", "Grade E - Critical")
        }

        for severity, (color, label) in severity_colors.items():
            symbol = QgsMarkerSymbol.createSimple({
                "name": "circle",
                "color": color,
                "size": "4"
            })
            category = QgsRendererCategory(severity, symbol, label)
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer("severity", categories)
        self.defect_layer.setRenderer(renderer)

        # Add to project
        QgsProject.instance().addMapLayer(self.defect_layer)

        # Zoom to layer
        self.iface.mapCanvas().setExtent(self.defect_layer.extent())
        self.iface.mapCanvas().refresh()

    def _update_statistics(self):
        """Update statistics labels."""
        if not self.defects:
            return

        self.stat_labels["total"].setText(str(len(self.defects)))
        self.stat_labels["potholes"].setText(
            str(sum(1 for d in self.defects if d.defect_type == "pothole")))
        self.stat_labels["alligator"].setText(
            str(sum(1 for d in self.defects if d.defect_type == "alligator_crack")))
        self.stat_labels["linear"].setText(
            str(sum(1 for d in self.defects if d.defect_type == "linear_crack")))
        self.stat_labels["grade_a"].setText(
            str(sum(1 for d in self.defects if d.severity == "A")))
        self.stat_labels["grade_b"].setText(
            str(sum(1 for d in self.defects if d.severity == "B")))
        self.stat_labels["grade_c"].setText(
            str(sum(1 for d in self.defects if d.severity == "C")))
        self.stat_labels["grade_d"].setText(
            str(sum(1 for d in self.defects if d.severity == "D")))
        self.stat_labels["grade_e"].setText(
            str(sum(1 for d in self.defects if d.severity == "E")))

    def _populate_defects_table(self, defects: List[DefectData] = None):
        """Populate defects table."""
        if defects is None:
            defects = self.defects

        self.defects_table.setRowCount(len(defects))

        for row, defect in enumerate(defects):
            self.defects_table.setItem(row, 0, QTableWidgetItem(str(defect.id)))
            self.defects_table.setItem(row, 1, QTableWidgetItem(defect.defect_type))

            severity_item = QTableWidgetItem(defect.severity)
            severity_colors = {"A": "#27ae60", "B": "#f1c40f", "C": "#e67e22",
                             "D": "#e74c3c", "E": "#8e44ad"}
            severity_item.setBackground(QColor(severity_colors.get(defect.severity, "#ffffff")))
            self.defects_table.setItem(row, 2, severity_item)

            self.defects_table.setItem(row, 3, QTableWidgetItem(f"{defect.confidence:.2f}"))
            self.defects_table.setItem(row, 4, QTableWidgetItem(f"{defect.x:.6f}"))
            self.defects_table.setItem(row, 5, QTableWidgetItem(f"{defect.y:.6f}"))
            self.defects_table.setItem(row, 6, QTableWidgetItem(f"{defect.estimated_cost:,.0f}"))

    def _filter_defects(self):
        """Filter defects table by severity and type."""
        severity = self.severity_filter.currentText()
        defect_type = self.type_filter.currentText()

        filtered = self.defects

        if severity != "All":
            filtered = [d for d in filtered if d.severity == severity]

        if defect_type != "All":
            filtered = [d for d in filtered if d.defect_type == defect_type]

        self._populate_defects_table(filtered)

    def _zoom_to_selected(self):
        """Zoom to selected defect in table."""
        selected = self.defects_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        x = float(self.defects_table.item(row, 4).text())
        y = float(self.defects_table.item(row, 5).text())

        # Zoom to point
        canvas = self.iface.mapCanvas()
        center = QgsPointXY(x, y)
        canvas.setCenter(center)
        canvas.zoomScale(1000)
        canvas.refresh()

    def _generate_route(self):
        """Generate optimal maintenance route."""
        if not self.defects:
            QMessageBox.warning(self, "Warning", "No defects detected. Run detection first.")
            return

        self.status_bar.setText("Generating optimal route...")
        QApplication.processEvents()

        # Get route based on selected algorithm
        algorithm = self.algorithm_combo.currentText()

        if self.priority_first.isChecked() or "Priority" in algorithm:
            self.route_order = TSPSolver.solve_with_priority(self.defects)
        else:
            points = [(d.x, d.y) for d in self.defects]
            self.route_order = TSPSolver.solve(points)

        # Calculate statistics
        points = [(self.defects[i].x, self.defects[i].y) for i in self.route_order]
        total_distance = TSPSolver.calculate_total_distance(
            [(d.x, d.y) for d in self.defects],
            self.route_order
        ) * 111000  # Convert to meters

        # Create route layer
        self._create_route_layer()

        # Display results
        high_priority = sum(1 for i in self.route_order
                          if self.defects[i].severity in ["D", "E"])
        total_cost = sum(self.defects[i].estimated_cost for i in self.route_order)

        result_text = f"""Route Generation Complete!

Algorithm: {algorithm}
Total Stops: {len(self.route_order)}
High Priority Stops (D, E): {high_priority}

Estimated Total Distance: {total_distance:.2f} meters
Total Repair Cost Estimate: {total_cost:,.0f} KRW

Route Order (by priority):
"""
        for idx, i in enumerate(self.route_order[:10], 1):
            d = self.defects[i]
            result_text += f"\n{idx}. ID:{d.id} - {d.defect_type} (Grade {d.severity})"

        if len(self.route_order) > 10:
            result_text += f"\n... and {len(self.route_order) - 10} more stops"

        self.route_result.setText(result_text)
        self.status_bar.setText("Route generation complete.")

    def _create_route_layer(self):
        """Create QGIS layer for route visualization."""
        if self.route_layer:
            QgsProject.instance().removeMapLayer(self.route_layer.id())

        self.route_layer = QgsVectorLayer(
            "LineString?crs=EPSG:4326",
            "Maintenance Route",
            "memory"
        )

        provider = self.route_layer.dataProvider()

        fields = QgsFields()
        fields.append(QgsField("segment", QVariant.Int))
        fields.append(QgsField("from_id", QVariant.Int))
        fields.append(QgsField("to_id", QVariant.Int))
        provider.addAttributes(fields)
        self.route_layer.updateFields()

        # Add line segments
        features = []
        for i in range(len(self.route_order) - 1):
            from_idx = self.route_order[i]
            to_idx = self.route_order[i + 1]

            from_point = QgsPointXY(self.defects[from_idx].x, self.defects[from_idx].y)
            to_point = QgsPointXY(self.defects[to_idx].x, self.defects[to_idx].y)

            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPolylineXY([from_point, to_point]))
            feature.setAttributes([i + 1, self.defects[from_idx].id, self.defects[to_idx].id])
            features.append(feature)

        provider.addFeatures(features)

        # Style
        symbol = QgsLineSymbol.createSimple({
            "color": "#3498db",
            "width": "1.5"
        })
        self.route_layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        QgsProject.instance().addMapLayer(self.route_layer)
        self.iface.mapCanvas().refresh()

    def _export_gpx(self):
        """Export route as GPX file."""
        if not self.route_order:
            QMessageBox.warning(self, "Warning", "No route generated. Generate route first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save GPX File", "", "GPX Files (*.gpx)"
        )

        if not file_path:
            return

        gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Road Pothole Detector">
  <trk>
    <name>Maintenance Route</name>
    <trkseg>
"""
        for i in self.route_order:
            d = self.defects[i]
            gpx_content += f'      <trkpt lat="{d.y}" lon="{d.x}"><name>ID:{d.id} - {d.defect_type} ({d.severity})</name></trkpt>\n'

        gpx_content += """    </trkseg>
  </trk>
</gpx>"""

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(gpx_content)

        QMessageBox.information(self, "Success", f"GPX file saved to:\n{file_path}")

    def _export_shapefile(self):
        """Export defects as shapefile."""
        if not self.defect_layer:
            QMessageBox.warning(self, "Warning", "No defect layer. Run detection first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Shapefile", "", "Shapefile (*.shp)"
        )

        if not file_path:
            return

        error = QgsVectorFileWriter.writeAsVectorFormat(
            self.defect_layer,
            file_path,
            "UTF-8",
            self.defect_layer.crs(),
            "ESRI Shapefile"
        )

        if error[0] == QgsVectorFileWriter.NoError:
            QMessageBox.information(self, "Success", f"Shapefile saved to:\n{file_path}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to save shapefile: {error[1]}")

    def _run_lcc_analysis(self):
        """Run Life Cycle Cost analysis."""
        if not self.defects:
            QMessageBox.warning(self, "Warning", "No defects detected. Run detection first.")
            return

        years = self.lcc_years.value()
        LCCAnalyzer.INFLATION_RATE = self.inflation_rate.value() / 100
        LCCAnalyzer.DISCOUNT_RATE = self.discount_rate.value() / 100

        results = LCCAnalyzer.analyze(self.defects, years)

        result_text = f"""Life Cycle Cost Analysis Results
================================

Analysis Period: {years} years
Inflation Rate: {self.inflation_rate.value():.1f}%
Discount Rate: {self.discount_rate.value():.1f}%

Cost Comparison:
----------------
Immediate Repair Cost: {results['immediate_cost']:,.0f} KRW
Delayed Repair Cost (NPV): {results['total_delayed_cost']:,.0f} KRW

Potential Savings: {results['savings']:,.0f} KRW ({results['savings_percent']:.1f}%)

Year-by-Year Delayed Cost (Cumulative NPV):
"""
        for i, cost in enumerate(results['delayed_costs'], 1):
            result_text += f"  Year {i}: {cost:,.0f} KRW\n"

        result_text += f"""
Recommendation:
---------------
{results['recommendation']}

Note: Immediate repair is typically more cost-effective as defect
deterioration accelerates over time, especially for higher severity grades.
"""

        self.lcc_result.setText(result_text)
        self.status_bar.setText("LCC analysis complete.")

    def _generate_report(self):
        """Generate work order report."""
        if not self.defects:
            QMessageBox.warning(self, "Warning", "No defects detected. Run detection first.")
            return

        report = f"""
================================================================================
                    ROAD MAINTENANCE WORK ORDER REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: Road Pothole Detector Plugin

================================================================================
                           EXECUTIVE SUMMARY
================================================================================
"""

        if self.include_stats.isChecked():
            total_cost = sum(d.estimated_cost for d in self.defects)
            report += f"""
DETECTION STATISTICS:
---------------------
Total Defects Detected: {len(self.defects)}

By Type:
  - Potholes: {sum(1 for d in self.defects if d.defect_type == 'pothole')}
  - Alligator Cracks: {sum(1 for d in self.defects if d.defect_type == 'alligator_crack')}
  - Linear Cracks: {sum(1 for d in self.defects if d.defect_type == 'linear_crack')}

By Severity:
  - Grade A (Minor): {sum(1 for d in self.defects if d.severity == 'A')}
  - Grade B: {sum(1 for d in self.defects if d.severity == 'B')}
  - Grade C: {sum(1 for d in self.defects if d.severity == 'C')}
  - Grade D: {sum(1 for d in self.defects if d.severity == 'D')}
  - Grade E (Critical): {sum(1 for d in self.defects if d.severity == 'E')}

Estimated Total Repair Cost: {total_cost:,.0f} KRW
"""

        if self.include_route.isChecked() and self.route_order:
            points = [(self.defects[i].x, self.defects[i].y) for i in self.route_order]
            distance = TSPSolver.calculate_total_distance(
                [(d.x, d.y) for d in self.defects], self.route_order) * 111000

            report += f"""
================================================================================
                        OPTIMIZED MAINTENANCE ROUTE
================================================================================
Algorithm Used: {self.algorithm_combo.currentText()}
Total Route Distance: {distance:.2f} meters
Number of Stops: {len(self.route_order)}

Route Sequence:
"""
            for idx, i in enumerate(self.route_order, 1):
                d = self.defects[i]
                report += f"  {idx:3d}. ID:{d.id:3d} | {d.defect_type:15s} | Grade {d.severity} | "
                report += f"({d.x:.6f}, {d.y:.6f}) | Cost: {d.estimated_cost:,.0f} KRW\n"

        if self.include_defect_list.isChecked():
            report += """
================================================================================
                          DETAILED DEFECT LIST
================================================================================
"""
            for d in sorted(self.defects, key=lambda x: x.priority):
                report += f"""
Defect ID: {d.id}
  Type: {d.defect_type}
  Severity: Grade {d.severity} (Priority: {d.priority})
  Location: ({d.x:.6f}, {d.y:.6f})
  Confidence: {d.confidence:.2%}
  Estimated Cost: {d.estimated_cost:,.0f} KRW
"""

        report += """
================================================================================
                              END OF REPORT
================================================================================
"""

        self.report_preview.setText(report)

        # Save if path specified
        if self.report_path.text():
            try:
                with open(self.report_path.text(), 'w', encoding='utf-8') as f:
                    f.write(report)
                QMessageBox.information(self, "Success",
                    f"Report saved to:\n{self.report_path.text()}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

        self.status_bar.setText("Report generated.")


class RoadPotholeDetector:
    """Main plugin class for Road Pothole Detector."""

    def __init__(self, iface):
        """Initialize plugin.

        :param iface: QGIS interface instance
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = PLUGIN_NAME
        self.toolbar = self.iface.addToolBar(PLUGIN_NAME)
        self.toolbar.setObjectName(PLUGIN_NAME)
        self.dialog = None

    def initGui(self):
        """Initialize GUI elements."""
        # Load icon
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            # Use standard icon as fallback
            icon = QApplication.style().standardIcon(QStyle.SP_DriveHDIcon)

        # Create action
        action = QAction(icon, PLUGIN_NAME, self.iface.mainWindow())
        action.triggered.connect(self.run)
        action.setEnabled(True)
        action.setStatusTip("Open Road Pothole Detector")

        # Add to toolbar and menu
        self.toolbar.addAction(action)
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)

    def unload(self):
        """Remove plugin menu items and icons."""
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

        # Remove toolbar
        if self.toolbar:
            del self.toolbar

    def run(self):
        """Run the plugin."""
        if self.dialog is None:
            self.dialog = MainDialog(self.iface, self.iface.mainWindow())

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
