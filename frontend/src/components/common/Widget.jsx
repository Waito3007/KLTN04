import React from 'react';
import styled from 'styled-components';
import theme from './theme';

const WidgetContainer = styled.div`
  background: ${theme.colors.white};
  border: 1px solid ${theme.colors.secondary};
  border-radius: 8px;
  padding: ${theme.spacing.md};
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: ${theme.spacing.sm};
`;

const WidgetTitle = styled.h3`
  font-size: ${theme.fontSizes.large};
  font-weight: 600;
  color: ${theme.colors.dark};
  margin: 0;
`;

const Widget = ({ title, children }) => {
  return (
    <WidgetContainer>
      <WidgetTitle>{title}</WidgetTitle>
      {children}
    </WidgetContainer>
  );
};

export default Widget;
