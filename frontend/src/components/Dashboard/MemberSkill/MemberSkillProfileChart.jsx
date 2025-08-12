import React, { useEffect, useState } from 'react';
import { Card, Typography, Empty, Select, Alert } from 'antd';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Loading } from '@components/common';

// Register Chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const { Title, Text } = Typography;
const { Option } = Select;

const MemberSkillProfileChart = ({ repositories = [], selectedRepoId }) => {
  const [skillProfiles, setSkillProfiles] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedMember, setSelectedMember] = useState('');

  // Fetch member skill profiles when repo changes
  useEffect(() => {
    if (selectedRepoId) {
      fetchMemberSkillProfiles();
    }
  }, [selectedRepoId]);

  const fetchMemberSkillProfiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`http://localhost:8000/api/member-skills/${selectedRepoId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch member skill profiles');
      }

      const data = await response.json();
      setSkillProfiles(data);
      
      // Auto-select first member if available
      const members = Object.keys(data);
      if (members.length > 0 && !selectedMember) {
        setSelectedMember(members[0]);
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching member skill profiles:', err);
    } finally {
      setLoading(false);
    }
  };

  // Generate radar chart data for selected member
  const generateRadarData = (memberProfile) => {
    if (!memberProfile) return null;

    // Calculate skill scores based on commit analysis
    const commitTypes = memberProfile.commit_types || {};
    const areas = memberProfile.areas || {};
    const totalCommits = memberProfile.total_commits || 1;

    // Skill categories for radar chart
    const skillCategories = {
      'Feature Development': (commitTypes.feat || 0) / totalCommits * 100,
      'Bug Fixing': (commitTypes.fix || 0) / totalCommits * 100,
      'Documentation': (commitTypes.docs || 0) / totalCommits * 100,
      'Refactoring': (commitTypes.refactor || 0) / totalCommits * 100,
      'Testing': (commitTypes.test || 0) / totalCommits * 100,
      'Frontend': (areas.frontend || 0) / totalCommits * 100,
      'Backend': (areas.backend || 0) / totalCommits * 100,
      'Database': (areas.database || 0) / totalCommits * 100,
    };

    return {
      labels: Object.keys(skillCategories),
      datasets: [
        {
          label: `${selectedMember} Skills`,
          data: Object.values(skillCategories),
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(54, 162, 235, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(54, 162, 235, 1)',
        },
      ],
    };
  };

  const radarOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Member Skill Profile',
      },
    },
    scales: {
      r: {
        angleLines: {
          display: true,
        },
        suggestedMin: 0,
        suggestedMax: 100,
        ticks: {
          stepSize: 20,
          callback: function(value) {
            return value + '%';
          },
        },
      },
    },
  };

  const members = Object.keys(skillProfiles);
  const selectedProfile = skillProfiles[selectedMember];
  const radarData = generateRadarData(selectedProfile);

  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Loading variant="circle" size="large" message="Loading member skill profiles..." />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  if (members.length === 0) {
    return (
      <Card>
        <Empty 
          description="No member skill data available"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      </Card>
    );
  }

  return (
    <Card>
      <div style={{ marginBottom: 16 }}>
        <Title level={4}>Member Skill Profiles</Title>
        <Text type="secondary">
          Skill analysis based on commit history and AI predictions
        </Text>
      </div>

      <div style={{ marginBottom: 16 }}>
        <Select
          style={{ width: 200 }}
          placeholder="Select a member"
          value={selectedMember}
          onChange={setSelectedMember}
        >
          {members.map(member => (
            <Option key={member} value={member}>
              {member}
            </Option>
          ))}
        </Select>
      </div>

      {selectedProfile && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Text strong>Profile Summary:</Text>
            <div style={{ marginTop: 8 }}>
              <Text>Total Commits: {selectedProfile.total_commits}</Text>
              <br />
              <Text>Risk Tolerance: {selectedProfile.risk_tolerance}</Text>
              <br />
              <Text>Expertise Areas: {selectedProfile.expertise_areas?.join(', ') || 'None'}</Text>
              <br />
              <Text>AI Coverage: {Math.round((selectedProfile.ai_coverage || 0) * 100)}%</Text>
            </div>
          </div>

          {radarData && (
            <div style={{ height: '400px', display: 'flex', justifyContent: 'center' }}>
              <Radar data={radarData} options={radarOptions} />
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

export default MemberSkillProfileChart;
