import React from 'react';

const Footer: React.FC = () => {
  return (
    <div 
      className="rounded-b-2xl"
      style={{
        background: '#ffffff',
        padding: '16px 9%',
        marginTop: '20px',
        borderTop: '1px solid #eaeaea',
        color: '#134f09'
      }}
    >
      <div 
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <div style={{ fontSize: '14px' }}>
          Smart Fertilizer Recommendation System
        </div>
        <div style={{ fontSize: '13px' }}>
          Created by Vidyashree
        </div>
      </div>
    </div>
  );
};

export default Footer;
