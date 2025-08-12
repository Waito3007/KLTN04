// Hook để sử dụng Toast với App context
import { App } from 'antd';

export const useToastContext = () => {
  const { message, notification } = App.useApp();
  return { message, notification };
};

export default useToastContext;
