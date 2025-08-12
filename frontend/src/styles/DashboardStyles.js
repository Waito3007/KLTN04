import styled from 'styled-components';

export const DashboardContainer = styled.div`
  padding: 24px;
  max-width: 1440px;
  margin: 0 auto;
  background: #f8fafc;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 24px;

  @media (max-width: 768px) {
    padding: 16px;
    gap: 16px;
  }
`;

export const MainLayout = styled.div`
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 24px;
  min-height: calc(100vh - 200px);

  @media (max-width: 1200px) {
    grid-template-columns: 250px 1fr;
    gap: 16px;
  }

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 16px;
  }
`;

export const Sidebar = styled.div`
  position: sticky;
  top: 24px;
  height: fit-content;
  display: flex;
  flex-direction: column;
  gap: 16px;

  @media (max-width: 768px) {
    position: static;
    order: 2;
  }
`;

export const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-width: 0; /* Để tránh overflow */
`;
