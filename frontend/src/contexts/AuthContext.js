import { createContext } from 'react';

/**
 * AuthContext để chia sẻ trạng thái xác thực trong toàn bộ ứng dụng
 * @type {React.Context<undefined>}
 */
export const AuthContext = createContext(undefined);
