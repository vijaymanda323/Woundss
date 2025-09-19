import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Dimensions,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import SimpleChart from './SimpleChart';

// Try to import react-native-chart-kit, fallback to SimpleChart if it fails
let LineChart, BarChart, PieChart;
try {
  const ChartKit = require('react-native-chart-kit');
  LineChart = ChartKit.LineChart;
  BarChart = ChartKit.BarChart;
  PieChart = ChartKit.PieChart;
} catch (error) {
  console.log('react-native-chart-kit not available, using SimpleChart fallback');
  LineChart = null;
  BarChart = null;
  PieChart = null;
}
import { getPatientHistory } from '../services/apiService';

const { width } = Dimensions.get('window');
const chartWidth = width - 40;

export default function WoundHealingGraphs({ patientId, currentAnalysis }) {
  const [patientHistory, setPatientHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedGraph, setSelectedGraph] = useState('area');

  useEffect(() => {
    loadPatientHistory();
  }, [patientId]);

  const loadPatientHistory = async () => {
    try {
      setIsLoading(true);
      const history = await getPatientHistory(patientId);
      setPatientHistory(history);
    } catch (error) {
      console.error('Error loading patient history:', error);
      // Use mock data for demonstration
      setPatientHistory(generateMockHistoryData());
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockHistoryData = () => {
    const mockData = [];
    const baseDate = new Date();
    const woundTypes = ['burn', 'cut', 'surgical', 'chronic'];
    const currentType = currentAnalysis?.woundType || 'burn';
    
    // Generate 7 days of mock data
    for (let i = 6; i >= 0; i--) {
      const date = new Date(baseDate);
      date.setDate(date.getDate() - i);
      
      // Simulate healing progression
      const baseArea = currentAnalysis?.area_cm2 || 5.0;
      const healingFactor = (7 - i) / 7; // Healing over time
      const area = Math.max(baseArea * (1 - healingFactor * 0.3), baseArea * 0.1);
      
      mockData.push({
        id: i + 1,
        date: date.toISOString(),
        area_cm2: parseFloat(area.toFixed(2)),
        wound_type: currentType,
        pain_level: Math.max(1, 5 - Math.floor(healingFactor * 4)),
        redness: ['severe', 'moderate', 'mild', 'none'][Math.floor(healingFactor * 4)],
        swelling: ['severe', 'moderate', 'mild', 'none'][Math.floor(healingFactor * 4)],
        healing_progress: Math.floor(healingFactor * 100),
      });
    }
    
    return mockData;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const getGraphData = () => {
    if (patientHistory.length === 0) return null;

    const sortedHistory = [...patientHistory].sort((a, b) => 
      new Date(a.date || a.timestamp) - new Date(b.date || b.timestamp)
    );

    const labels = sortedHistory.map(item => 
      formatDate(item.date || item.timestamp)
    );

    const areaData = sortedHistory.map(item => {
      const value = item.area_cm2 || item.area || 0;
      return typeof value === 'number' && !isNaN(value) ? value : 0;
    });
    
    const painData = sortedHistory.map(item => {
      const value = item.pain_level || item.painLevel || 0;
      return typeof value === 'number' && !isNaN(value) ? value : 0;
    });
    
    const progressData = sortedHistory.map(item => {
      const value = item.healing_progress || item.healingProgress || 0;
      return typeof value === 'number' && !isNaN(value) ? value : 0;
    });

    return {
      labels,
      areaData,
      painData,
      progressData,
    };
  };

  const getWoundTypeDistribution = () => {
    const typeCount = {};
    patientHistory.forEach(item => {
      const type = item.wound_type || item.predicted_label || 'unknown';
      typeCount[type] = (typeCount[type] || 0) + 1;
    });

    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    const pieData = Object.entries(typeCount).map(([type, count], index) => ({
      name: type.charAt(0).toUpperCase() + type.slice(1),
      population: count,
      color: colors[index % colors.length],
      legendFontColor: '#7F7F7F',
      legendFontSize: 12,
    }));

    return pieData;
  };

  const chartConfig = {
    backgroundColor: '#ffffff',
    backgroundGradientFrom: '#ffffff',
    backgroundGradientTo: '#ffffff',
    decimalPlaces: 1,
    color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#667eea',
    },
  };

  const renderAreaChart = () => {
    const graphData = getGraphData();
    if (!graphData || graphData.areaData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No area data available</Text>
        </View>
      );
    }

    // Validate data before rendering
    const validData = graphData.areaData.filter(value => 
      typeof value === 'number' && !isNaN(value) && value >= 0
    );
    
    if (validData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No valid area data available</Text>
        </View>
      );
    }

    if (LineChart) {
      try {
        return (
          <LineChart
            data={{
              labels: graphData.labels.slice(0, validData.length),
              datasets: [
                {
                  data: validData,
                  color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
                  strokeWidth: 3,
                },
              ],
            }}
            width={chartWidth}
            height={220}
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
            withDots={true}
            withShadow={false}
            withScrollableDot={false}
          />
        );
      } catch (error) {
        console.error('LineChart error:', error);
        // Fallback to SimpleChart on error
        const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
          label: label,
          value: validData[index],
        }));
        return (
          <SimpleChart
            data={chartData}
            title="Wound Area Progression (cm²)"
            type="line"
            color="#667eea"
          />
        );
      }
    } else {
      // Fallback to SimpleChart
      const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
        label: label,
        value: validData[index],
      }));
      return (
        <SimpleChart
          data={chartData}
          title="Wound Area Progression (cm²)"
          type="line"
          color="#667eea"
        />
      );
    }
  };

  const renderPainChart = () => {
    const graphData = getGraphData();
    if (!graphData || graphData.painData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No pain data available</Text>
        </View>
      );
    }

    // Validate data before rendering
    const validData = graphData.painData.filter(value => 
      typeof value === 'number' && !isNaN(value) && value >= 0
    );
    
    if (validData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No valid pain data available</Text>
        </View>
      );
    }

    if (BarChart) {
      try {
        return (
          <BarChart
            data={{
              labels: graphData.labels.slice(0, validData.length),
              datasets: [
                {
                  data: validData,
                },
              ],
            }}
            width={chartWidth}
            height={220}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(231, 76, 60, ${opacity})`,
            }}
            style={styles.chart}
            withHorizontalLabels={true}
            withVerticalLabels={true}
            showValuesOnTopOfBars={true}
          />
        );
      } catch (error) {
        console.error('BarChart error:', error);
        // Fallback to SimpleChart on error
        const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
          label: label,
          value: validData[index],
        }));
        return (
          <SimpleChart
            data={chartData}
            title="Pain Level Over Time (1-5 Scale)"
            type="bar"
            color="#e74c3c"
          />
        );
      }
    } else {
      // Fallback to SimpleChart
      const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
        label: label,
        value: validData[index],
      }));
      return (
        <SimpleChart
          data={chartData}
          title="Pain Level Over Time (1-5 Scale)"
          type="bar"
          color="#e74c3c"
        />
      );
    }
  };

  const renderProgressChart = () => {
    const graphData = getGraphData();
    if (!graphData || graphData.progressData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No progress data available</Text>
        </View>
      );
    }

    // Validate data before rendering
    const validData = graphData.progressData.filter(value => 
      typeof value === 'number' && !isNaN(value) && value >= 0
    );
    
    if (validData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No valid progress data available</Text>
        </View>
      );
    }

    if (LineChart) {
      try {
        return (
          <LineChart
            data={{
              labels: graphData.labels.slice(0, validData.length),
              datasets: [
                {
                  data: validData,
                  color: (opacity = 1) => `rgba(39, 174, 96, ${opacity})`,
                  strokeWidth: 3,
                },
              ],
            }}
            width={chartWidth}
            height={220}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(39, 174, 96, ${opacity})`,
            }}
            style={styles.chart}
            withDots={true}
            withShadow={false}
            withScrollableDot={false}
          />
        );
      } catch (error) {
        console.error('LineChart error:', error);
        // Fallback to SimpleChart on error
        const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
          label: label,
          value: validData[index],
        }));
        return (
          <SimpleChart
            data={chartData}
            title="Healing Progress (%)"
            type="line"
            color="#27ae60"
          />
        );
      }
    } else {
      // Fallback to SimpleChart
      const chartData = graphData.labels.slice(0, validData.length).map((label, index) => ({
        label: label,
        value: validData[index],
      }));
      return (
        <SimpleChart
          data={chartData}
          title="Healing Progress (%)"
          type="line"
          color="#27ae60"
        />
      );
    }
  };

  const renderWoundTypeChart = () => {
    const pieData = getWoundTypeDistribution();
    if (pieData.length === 0) {
      return (
        <View style={styles.noDataContainer}>
          <Text style={styles.noDataText}>No wound type data available</Text>
        </View>
      );
    }

    if (PieChart) {
      try {
        return (
          <PieChart
            data={pieData}
            width={chartWidth}
            height={220}
            chartConfig={chartConfig}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            style={styles.chart}
            center={[10, 0]}
            absolute
          />
        );
      } catch (error) {
        console.error('PieChart error:', error);
        // Fallback to SimpleChart on error
        const chartData = pieData.map(item => ({
          label: item.name,
          value: item.population,
          color: item.color,
        }));
        return (
          <SimpleChart
            data={chartData}
            title="Wound Type Distribution"
            type="pie"
            color="#667eea"
          />
        );
      }
    } else {
      // Fallback to SimpleChart
      const chartData = pieData.map(item => ({
        label: item.name,
        value: item.population,
        color: item.color,
      }));
      return (
        <SimpleChart
          data={chartData}
          title="Wound Type Distribution"
          type="pie"
          color="#667eea"
        />
      );
    }
  };

  const renderCurrentGraph = () => {
    switch (selectedGraph) {
      case 'area':
        return renderAreaChart();
      case 'pain':
        return renderPainChart();
      case 'progress':
        return renderProgressChart();
      case 'types':
        return renderWoundTypeChart();
      default:
        return renderAreaChart();
    }
  };

  const getGraphTitle = () => {
    switch (selectedGraph) {
      case 'area':
        return 'Wound Area Progression (cm²)';
      case 'pain':
        return 'Pain Level Over Time (1-5 Scale)';
      case 'progress':
        return 'Healing Progress (%)';
      case 'types':
        return 'Wound Type Distribution';
      default:
        return 'Wound Area Progression (cm²)';
    }
  };

  const getGraphDescription = () => {
    switch (selectedGraph) {
      case 'area':
        return 'Track how the wound area changes over time. Decreasing area indicates healing progress.';
      case 'pain':
        return 'Monitor pain levels to assess healing and treatment effectiveness.';
      case 'progress':
        return 'Overall healing progress percentage based on multiple factors.';
      case 'types':
        return 'Distribution of different wound types in your medical history.';
      default:
        return 'Track how the wound area changes over time.';
    }
  };

  if (isLoading) {
    return (
      <Card style={styles.container}>
        <Card.Content>
          <View style={styles.loadingContainer}>
            <Ionicons name="analytics-outline" size={48} color="#667eea" />
            <Title style={styles.loadingTitle}>Loading Healing Graphs...</Title>
            <Paragraph style={styles.loadingText}>
              Analyzing your wound healing progress
            </Paragraph>
          </View>
        </Card.Content>
      </Card>
    );
  }

  return (
    <Card style={styles.container}>
      <Card.Content>
        <View style={styles.header}>
          <View style={styles.titleContainer}>
            <Ionicons name="analytics" size={24} color="#667eea" />
            <Title style={styles.title}>Wound Healing Analytics</Title>
          </View>
          <Chip style={styles.patientChip} textStyle={styles.chipText}>
            Patient: {patientId}
          </Chip>
        </View>

        <View style={styles.graphSelector}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View style={styles.selectorContainer}>
              {[
                { key: 'area', label: 'Area', icon: 'resize' },
                { key: 'pain', label: 'Pain', icon: 'heart-outline' },
                { key: 'progress', label: 'Progress', icon: 'trending-up' },
                { key: 'types', label: 'Types', icon: 'chart-pie' },
              ].map((item) => (
                <Button
                  key={item.key}
                  mode={selectedGraph === item.key ? 'contained' : 'outlined'}
                  onPress={() => setSelectedGraph(item.key)}
                  style={[
                    styles.selectorButton,
                    selectedGraph === item.key && styles.selectedButton,
                  ]}
                  icon={item.icon}
                  compact
                >
                  {item.label}
                </Button>
              ))}
            </View>
          </ScrollView>
        </View>

        <View style={styles.graphContainer}>
          <View style={styles.graphHeader}>
            <Title style={styles.graphTitle}>{getGraphTitle()}</Title>
            <Paragraph style={styles.graphDescription}>
              {getGraphDescription()}
            </Paragraph>
          </View>
          
          {renderCurrentGraph()}
        </View>

        <View style={styles.summaryContainer}>
          <Title style={styles.summaryTitle}>Summary</Title>
          <View style={styles.summaryRow}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Total Records</Text>
              <Text style={styles.summaryValue}>{patientHistory.length}</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Current Area</Text>
              <Text style={styles.summaryValue}>
                {currentAnalysis?.area_cm2 || 'N/A'} cm²
              </Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Healing Time</Text>
              <Text style={styles.summaryValue}>
                {currentAnalysis?.estimated_days_to_cure || 'N/A'} days
              </Text>
            </View>
          </View>
        </View>
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: {
    margin: 15,
    elevation: 4,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 8,
    color: '#2c3e50',
  },
  patientChip: {
    backgroundColor: '#667eea',
  },
  chipText: {
    color: 'white',
    fontSize: 12,
  },
  graphSelector: {
    marginBottom: 20,
  },
  selectorContainer: {
    flexDirection: 'row',
    paddingHorizontal: 5,
  },
  selectorButton: {
    marginRight: 10,
    borderRadius: 20,
  },
  selectedButton: {
    backgroundColor: '#667eea',
  },
  graphContainer: {
    marginBottom: 20,
  },
  graphHeader: {
    marginBottom: 15,
  },
  graphTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  graphDescription: {
    fontSize: 14,
    color: '#7f8c8d',
    lineHeight: 20,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  summaryContainer: {
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 12,
    marginTop: 10,
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  summaryItem: {
    alignItems: 'center',
    flex: 1,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 40,
  },
  loadingTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginTop: 15,
    marginBottom: 10,
  },
  loadingText: {
    fontSize: 14,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  noDataContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    marginVertical: 10,
  },
  noDataText: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
