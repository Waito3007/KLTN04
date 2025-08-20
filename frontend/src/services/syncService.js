// e:\Project\KLTN04-1\frontend\src\services\syncService.js
export const getSyncStatus = async () => {
  const response = await fetch('/api/sync-status');
  return response.json();
};

export const syncRepository = async (repoKey) => {
  const response = await fetch(`/api/sync/${repoKey}`, { method: 'POST' });
  return response.json();
};

export const getRepoEvents = async (owner, name) => {
  const response = await fetch(`/api/events/${owner}/${name}`);
  return response.json();
};
