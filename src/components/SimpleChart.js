import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { Card, Title, Paragraph } from 'react-native-paper';

const { width } = Dimensions.get('window');
const chartWidth = width - 60;

export default function SimpleChart({ data, title, type = 'line', color = '#667eea' }) {
  const maxValue = Math.max(...data.map(item => item.value));
  const minValue = Math.min(...data.map(item => item.value));
  const range = maxValue - minValue;

  const renderLineChart = () => {
    const points = data.map((item, index) => {
      const x = (index / (data.length - 1)) * (chartWidth - 40);
      const y = 120 - ((item.value - minValue) / range) * 100;
      return { x, y };
    });

    return (
      <View style={styles.chartContainer}>
        <View style={styles.chartArea}>
          {/* Y-axis labels */}
          <View style={styles.yAxis}>
            <Text style={styles.axisLabel}>{maxValue.toFixed(1)}</Text>
            <Text style={styles.axisLabel}>{((maxValue + minValue) / 2).toFixed(1)}</Text>
            <Text style={styles.axisLabel}>{minValue.toFixed(1)}</Text>
          </View>
          
          {/* Chart area */}
          <View style={styles.chartArea}>
            {/* Grid lines */}
            <View style={styles.gridLine} />
            <View style={[styles.gridLine, { top: 60 }]} />
            <View style={[styles.gridLine, { top: 120 }]} />
            
            {/* Data points and lines */}
            {points.map((point, index) => (
              <View key={index}>
                {/* Line to next point */}
                {index < points.length - 1 && (
                  <View
                    style={[
                      styles.line,
                      {
                        left: point.x + 10,
                        top: point.y + 10,
                        width: Math.sqrt(
                          Math.pow(points[index + 1].x - point.x, 2) +
                          Math.pow(points[index + 1].y - point.y, 2)
                        ),
                        transform: [
                          {
                            rotate: `${Math.atan2(
                              points[index + 1].y - point.y,
                              points[index + 1].x - point.x
                            )}rad`,
                          },
                        ],
                      },
                    ]}
                  />
                )}
                
                {/* Data point */}
                <View
                  style={[
                    styles.dataPoint,
                    {
                      left: point.x + 5,
                      top: point.y + 5,
                      backgroundColor: color,
                    },
                  ]}
                />
                
                {/* Value label */}
                <Text
                  style={[
                    styles.valueLabel,
                    {
                      left: point.x,
                      top: point.y - 20,
                    },
                  ]}
                >
                  {data[index].value.toFixed(1)}
                </Text>
              </View>
            ))}
          </View>
        </View>
        
        {/* X-axis labels */}
        <View style={styles.xAxis}>
          {data.map((item, index) => (
            <Text key={index} style={styles.xAxisLabel}>
              {item.label}
            </Text>
          ))}
        </View>
      </View>
    );
  };

  const renderBarChart = () => {
    return (
      <View style={styles.chartContainer}>
        <View style={styles.barChartArea}>
          {/* Y-axis labels */}
          <View style={styles.yAxis}>
            <Text style={styles.axisLabel}>{maxValue.toFixed(1)}</Text>
            <Text style={styles.axisLabel}>{((maxValue + minValue) / 2).toFixed(1)}</Text>
            <Text style={styles.axisLabel}>{minValue.toFixed(1)}</Text>
          </View>
          
          {/* Bars */}
          <View style={styles.barsContainer}>
            {data.map((item, index) => {
              const height = ((item.value - minValue) / range) * 100;
              return (
                <View key={index} style={styles.barGroup}>
                  <View
                    style={[
                      styles.bar,
                      {
                        height: height,
                        backgroundColor: color,
                      },
                    ]}
                  />
                  <Text style={styles.barValue}>{item.value.toFixed(1)}</Text>
                  <Text style={styles.barLabel}>{item.label}</Text>
                </View>
              );
            })}
          </View>
        </View>
      </View>
    );
  };

  const renderPieChart = () => {
    const total = data.reduce((sum, item) => sum + item.value, 0);
    let currentAngle = 0;
    
    return (
      <View style={styles.chartContainer}>
        <View style={styles.pieChartArea}>
          <View style={styles.pieChart}>
            {data.map((item, index) => {
              const percentage = (item.value / total) * 100;
              const angle = (item.value / total) * 360;
              const startAngle = currentAngle;
              currentAngle += angle;
              
              return (
                <View key={index} style={styles.pieSlice}>
                  <View
                    style={[
                      styles.pieSegment,
                      {
                        backgroundColor: item.color || color,
                        transform: [{ rotate: `${startAngle}deg` }],
                      },
                    ]}
                  />
                  <Text style={styles.pieLabel}>
                    {item.label}: {percentage.toFixed(1)}%
                  </Text>
                </View>
              );
            })}
          </View>
        </View>
      </View>
    );
  };

  const renderChart = () => {
    switch (type) {
      case 'bar':
        return renderBarChart();
      case 'pie':
        return renderPieChart();
      default:
        return renderLineChart();
    }
  };

  return (
    <Card style={styles.container}>
      <Card.Content>
        <Title style={styles.title}>{title}</Title>
        {renderChart()}
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: {
    margin: 15,
    elevation: 4,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
    textAlign: 'center',
  },
  chartContainer: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  chartArea: {
    position: 'relative',
    width: chartWidth,
    height: 140,
    marginBottom: 20,
  },
  yAxis: {
    position: 'absolute',
    left: 0,
    top: 0,
    height: 120,
    justifyContent: 'space-between',
    width: 30,
  },
  axisLabel: {
    fontSize: 10,
    color: '#7f8c8d',
    textAlign: 'right',
  },
  gridLine: {
    position: 'absolute',
    left: 30,
    right: 10,
    height: 1,
    backgroundColor: '#ecf0f1',
  },
  line: {
    position: 'absolute',
    height: 2,
    backgroundColor: '#667eea',
  },
  dataPoint: {
    position: 'absolute',
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: '#ffffff',
  },
  valueLabel: {
    position: 'absolute',
    fontSize: 10,
    color: '#2c3e50',
    fontWeight: 'bold',
    textAlign: 'center',
    width: 20,
  },
  xAxis: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: chartWidth,
    paddingHorizontal: 30,
  },
  xAxisLabel: {
    fontSize: 10,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  barChartArea: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 140,
    width: chartWidth,
  },
  barsContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-around',
    flex: 1,
    marginLeft: 30,
  },
  barGroup: {
    alignItems: 'center',
    flex: 1,
  },
  bar: {
    width: 20,
    marginBottom: 5,
    borderRadius: 2,
  },
  barValue: {
    fontSize: 10,
    color: '#2c3e50',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  barLabel: {
    fontSize: 10,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  pieChartArea: {
    alignItems: 'center',
  },
  pieChart: {
    width: 120,
    height: 120,
    borderRadius: 60,
    overflow: 'hidden',
    marginBottom: 20,
  },
  pieSlice: {
    position: 'absolute',
    width: 120,
    height: 120,
  },
  pieSegment: {
    width: 120,
    height: 120,
    borderRadius: 60,
  },
  pieLabel: {
    fontSize: 12,
    color: '#2c3e50',
    textAlign: 'center',
    marginTop: 10,
  },
});

