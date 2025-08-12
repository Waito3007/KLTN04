// theme.js - Enhanced Design System

const theme = {
  colors: {
    primary: "#1890ff",
    primaryDark: "#0050b3",
    primaryLight: "#69c0ff",
    secondary: "#6C757D",
    success: "#52c41a",
    danger: "#ff4d4f",
    warning: "#faad14",
    info: "#13c2c2",
    light: "#f0f2f5",
    dark: "#001529",
    white: "#FFFFFF",
    black: "#000000",
    
    // AI/Tech theme colors
    techBlue: "#667eea",
    techPurple: "#764ba2",
    techGreen: "#00b96b",
    techCyan: "#13c2c2",
    techOrange: "#fa8c16",
    
    // Gradient system
    gradient: {
      primary: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      tech: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      success: "linear-gradient(135deg, #52c41a 0%, #00b96b 100%)",
      warning: "linear-gradient(135deg, #faad14 0%, #fa8c16 100%)",
      danger: "linear-gradient(135deg, #ff4d4f 0%, #cf1322 100%)",
      info: "linear-gradient(135deg, #1890ff 0%, #13c2c2 100%)",
      dark: "linear-gradient(135deg, #001529 0%, #343a40 100%)",
      glass: "linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)",
      subtle: "linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)",
    },
    
    // Background colors
    bg: {
      primary: "#ffffff",
      secondary: "#fafafa",
      tertiary: "#f0f2f5",
      dark: "#141414",
      glass: "rgba(255, 255, 255, 0.95)",
    },

    // Text colors
    text: {
      primary: "#262626",
      secondary: "#595959",
      tertiary: "#8c8c8c",
      inverse: "#ffffff",
      disabled: "#bfbfbf",
    },

    // Border colors
    border: {
      light: "#f0f0f0",
      default: "#d9d9d9",
      dark: "#434343",
    }
  },

  // Spacing system
  spacing: {
    xs: "4px",
    sm: "8px",
    md: "16px",
    lg: "24px",
    xl: "32px",
    xxl: "48px",
    xxxl: "64px",
  },

  // Typography
  fontSizes: {
    xs: "12px",
    sm: "14px",
    md: "16px",
    lg: "18px",
    xl: "20px",
    xxl: "24px",
    xxxl: "32px",
  },

  fontWeights: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // Border radius
  borderRadius: {
    sm: "6px",
    md: "8px",
    lg: "12px",
    xl: "16px",
    xxl: "24px",
    round: "50%",
  },

  // Shadows system
  shadows: {
    sm: "0 1px 2px rgba(0, 0, 0, 0.05)",
    md: "0 4px 6px rgba(0, 0, 0, 0.07)",
    lg: "0 10px 15px rgba(0, 0, 0, 0.1)",
    xl: "0 20px 25px rgba(0, 0, 0, 0.15)",
    glow: "0 0 20px rgba(102, 126, 234, 0.3)",
    glowHover: "0 0 30px rgba(102, 126, 234, 0.4)",
    glass: "0 8px 32px rgba(31, 38, 135, 0.37)",
    modern: "0 8px 24px rgba(0, 0, 0, 0.12)",
  },

  // Animation
  transitions: {
    fast: "0.15s cubic-bezier(0.4, 0, 0.2, 1)",
    normal: "0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    slow: "0.5s cubic-bezier(0.4, 0, 0.2, 1)",
  },

  // Z-index system
  zIndex: {
    dropdown: 1000,
    modal: 1050,
    popover: 1060,
    tooltip: 1070,
    notification: 1080,
  },

  // Breakpoints
  breakpoints: {
    xs: "480px",
    sm: "576px", 
    md: "768px",
    lg: "992px",
    xl: "1200px",
    xxl: "1600px",
  }
};

export default theme;
