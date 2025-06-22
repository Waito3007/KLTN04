// kanbanConstants.js
import { ClockCircleOutlined, ExclamationCircleOutlined, CheckCircleOutlined } from '@ant-design/icons';

export const COLUMN_CONFIG = [
  {
    id: 'TODO',
    title: 'To Do',
    icon: ClockCircleOutlined,
    color: '#faad14',
    bgColor: '#faad14',
    borderColor: '#faad14',
    cssClass: 'todo'
  },
  {
    id: 'IN_PROGRESS',
    title: 'In Progress',
    icon: ExclamationCircleOutlined,
    color: '#1890ff',
    bgColor: '#1890ff',
    borderColor: '#1890ff',
    cssClass: 'inProgress'
  },
  {
    id: 'DONE',
    title: 'Done',
    icon: CheckCircleOutlined,
    color: '#52c41a',
    bgColor: '#52c41a',
    borderColor: '#52c41a',
    cssClass: 'done'
  }
];

export const DRAG_CONFIG = {
  ACTIVATION_DISTANCE: 0,
  DROP_ANIMATION: {
    duration: 200,
    easing: 'cubic-bezier(0.18, 0.67, 0.6, 1.22)',
  }
};

export const TASK_CARD_CONFIG = {
  DESCRIPTION_MAX_LENGTH: 35,
  AVATAR_SIZE: 24
};
