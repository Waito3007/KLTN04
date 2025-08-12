import React from 'react';
import styled from 'styled-components';
import theme from './theme';

const FooterContainer = styled.footer`
  background: ${theme.colors.dark};
  color: ${theme.colors.white};
  text-align: center;
  padding: ${theme.spacing.md} 0;
  position: fixed;
  bottom: 0;
  width: 100%;
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
`;

const Footer = () => {
  return (
    <FooterContainer>
      <p>© 2025 KLTN04. All rights reserved.</p>
    </FooterContainer>
  );
};

export default Footer;
