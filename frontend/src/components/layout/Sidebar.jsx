import React from 'react';
import { NavLink } from 'react-router-dom';
import { LuLayoutDashboard, LuGitBranch, LuBarChart, LuSettings } from "react-icons/lu";

const Sidebar = () => {
  return (
    <div className="w-64 bg-gray-800 text-white flex flex-col">
      <div className="p-4 text-2xl font-bold text-center border-b border-gray-700">YourLogo</div>
      <nav className="flex-1 px-2 py-4 space-y-2">
        <NavLink to="/dashboard" className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white rounded-md">
          <LuLayoutDashboard className="mr-3" />
          Dashboard
        </NavLink>
        <NavLink to="/repo-sync" className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white rounded-md">
          <LuGitBranch className="mr-3" />
          Repositories
        </NavLink>
        <NavLink to="/analysis" className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white rounded-md">
          <LuBarChart className="mr-3" />
          Analysis
        </NavLink>
        <NavLink to="/settings" className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white rounded-md">
          <LuSettings className="mr-3" />
          Settings
        </NavLink>
      </nav>
    </div>
  );
};

export default Sidebar;
